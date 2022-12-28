mod transform;

use opencv::core::*;
use opencv::dnn::DNN_BACKEND_OPENCV;
use opencv::dnn::DNN_TARGET_CPU;
use opencv::dnn::prelude::*;
use opencv::dnn::read_net_from_tensorflow;
use opencv::dnn::Net;
use opencv::imgcodecs::imread;
use opencv::imgproc::*;
use opencv::highgui::{imshow, wait_key};
use opencv::types::VectorOfVectorOfPoint;

use crate::transform::four_point_transform;

fn find_puzzle(img: &Mat) -> Option<(Mat, Mat)> {
    let mut gray = Mat::default();
    cvt_color(&img, &mut gray, COLOR_BGR2GRAY, 0)
        .expect("Could not convert image to grayscale");

    let mut blurred = Mat::default();
    gaussian_blur(&gray, 
                  &mut blurred, 
                  opencv::core::Size::new(5, 5), 
                  0.0, 0.0, 
                  opencv::core::BORDER_DEFAULT)
        .expect("Could not blur image");

    let mut thresh = Mat::default();
    adaptive_threshold(&blurred, 
                       &mut thresh, 
                       255.0, 
                       ADAPTIVE_THRESH_GAUSSIAN_C, 
                       THRESH_BINARY, 
                       11, 2.0)
        .expect("Could not threshold image");

    let mut inverted = Mat::default();
    bitwise_not(&thresh, &mut inverted, &opencv::core::no_array())
        .expect("Could not invert image");

    let mut contours = opencv::types::VectorOfVectorOfPoint::new();
    find_contours(&inverted, 
                  &mut contours,
                  RETR_EXTERNAL, 
                  CHAIN_APPROX_SIMPLE, 
                  Point::new(0, 0))
        .expect("Could not find contours");

    let mut puzzle_cnt = Mat::default();
    let mut max_peri = 0.0;

    for c in &contours {
        let peri = arc_length(&c, true).expect("Could not calculate arc length");
        let mut approx = Mat::default();
        approx_poly_dp(&c, &mut approx, 0.02 * peri, true)
            .expect("Could not approximate polygon");
        if approx.rows() == 4 && peri > max_peri {
            puzzle_cnt = approx;
            max_peri = peri;
        }
    }

    if puzzle_cnt.rows() != 4 {
        panic!("Could not find puzzle");
    }

    let puzzle = four_point_transform(&img, &puzzle_cnt);
    let warped = four_point_transform(&gray, &puzzle_cnt);

    Some((puzzle, warped))
}

fn extract_digit(cell: &Mat) -> Option<Mat> {
    let thresh = clear_border(cell);
    let mut cnts = VectorOfVectorOfPoint::new();
    let ret = find_contours(&thresh, 
                  &mut cnts, 
                  RETR_EXTERNAL, 
                  CHAIN_APPROX_SIMPLE, 
                  Point::new(0, 0));
    if ret.is_err() || cnts.len() == 0 {
        return None;
    }
    return Some(thresh);
}

fn clear_border(img: &Mat) -> Mat {
    let mut norm = Mat::default();
    normalize(&img, 
              &mut norm, 
              0.0, 255.0, 
              NORM_MINMAX, 
              CV_8U, 
              &no_array())
        .expect("Could not normalize image");

    let mut thresh = Mat::default();
    threshold(&norm, 
              &mut thresh, 
              0.0, 
              255.0, 
              THRESH_BINARY_INV | THRESH_OTSU)
        .expect("Could not threshold image");

    let mut contours = VectorOfVectorOfPoint::new();
    find_contours(&thresh, 
                  &mut contours, 
                  RETR_EXTERNAL,
                  CHAIN_APPROX_SIMPLE, 
                  Point::new(0, 0))
        .expect("Could not find contours");

  
    let mut mask = Mat::ones_size(
        img.size().expect("Could not get image size"), 
        CV_8U)
        .expect("Could not create mask")
        .to_mat()
        .expect("Could not convert mask to Mat");

    for c in &contours {
        let rect = bounding_rect(&c).expect("Could not get bounding rect");
        let tl = rect.x;
        let tr = rect.x + rect.width;
        let bl = rect.y;
        let br = rect.y + rect.height;
        if tl == 0 || tr == img.cols() || bl == 0 || br == img.rows() {
            fill_poly(&mut mask, &c, Scalar::all(0.0), LINE_8, 0, Point::new(0, 0))
                .expect("Could not fill polygon");
        }
    }
    
    let mut masked = Mat::default();
    bitwise_and(&thresh, &thresh, &mut masked, &mask)
        .expect("Could not apply mask");

    return masked;
}

fn predict(model: &mut Net, cell: &Mat) -> u8 {
    let mut out = Mat::default();
    resize(&cell, &mut out, opencv::core::Size::new(28, 28), 0.0, 0.0, INTER_AREA)
        .expect("Could not resize image");

    return 1;
}

fn print_board(board: &[[u8; 9]; 9]) {
    for row in board {
        for col in row {
            print!("{} ", col);
        }
        println!();
    }
}

fn main() {

    let img = imread("res/img/sudoku0.jpg", opencv::imgcodecs::IMREAD_COLOR)
        .expect("Could not read image");

    let mut out = Mat::default();
    resize(&img, 
           &mut out,
           Size::new(600, 600),
           0.0, 0.0,
           INTER_AREA)
        .expect("Could not resize image");


    let (puzzle, warped) = find_puzzle(&out).expect("Could not find puzzle");

    let mut board = [[0; 9]; 9];

    let step_x = warped.cols() / 9;
    let step_y = warped.rows() / 9;

    let mut model = read_net_from_tensorflow("res/model/frozen_graph.pb", "")
        .expect("Could not read model");
    model.set_preferable_backend(DNN_BACKEND_OPENCV)
        .expect("Could not set backend");
    model.set_preferable_target(DNN_TARGET_CPU)
        .expect("Could not set target");

    for y in 0..9 {
        for x in 0..9 {
            let start_x = x * step_x;
            let start_y = y * step_y;

            let mut cell = Mat::default();
            let roi = Rect::new(start_x, start_y, step_x, step_y);
            Mat::roi(&warped, roi)
                .expect("Could not get ROI")
                .copy_to(&mut cell)
                .expect("Could not copy ROI to cell");

            let digit = extract_digit(&cell);
            let val = match digit {
                Some(digit) => predict(&mut model, &digit),
                None => 0,
            };

            board[y as usize][x as usize] = val;
        }
    }

    print_board(&board);

    imshow("Display window", &puzzle).expect("Could not show image");
    wait_key(0).expect("Could not wait for key");
}
