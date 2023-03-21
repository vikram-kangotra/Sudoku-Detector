mod transform;

use opencv::core::*;
use opencv::imgcodecs::imread;
use opencv::imgproc::*;
use opencv::highgui::{imshow, wait_key};
use opencv::types::VectorOfVectorOfPoint;
use tract_tensorflow::prelude::*;

use crate::transform::four_point_transform;

fn preprocess_image(img: &Mat) -> Mat {
    let mut blurred = Mat::default();
    let blur_radius = 9;
    gaussian_blur(&img, 
                  &mut blurred, 
                  opencv::core::Size::new(blur_radius, blur_radius), 
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
    
    thresh
}

fn find_puzzle(img: &Mat) -> Option<Mat> {
    let processed =  preprocess_image(img);

    let mut inverted = Mat::default();
    bitwise_not(&processed, &mut inverted, &opencv::core::no_array())
        .expect("Could not invert image");

    let mut contours = VectorOfVectorOfPoint::new();
    find_contours(&inverted, 
                  &mut contours,
                  RETR_EXTERNAL, 
                  CHAIN_APPROX_SIMPLE, 
                  Point::new(0, 0))
        .expect("Could not find contours");

    let mut corner_pts = Mat::default();
    let mut max_perimeter = 0.0;

    for c in &contours {
        let perimeter = arc_length(&c, true).expect("Could not calculate arc length");
        let mut approx = Mat::default();
        approx_poly_dp(&c,
                       &mut approx,
                       0.02 * perimeter, 
                       true)
            .expect("Could not approximate polygon");
        
        if approx.rows() == 4 && perimeter > max_perimeter {
            corner_pts = approx;
            max_perimeter = perimeter;
        }
    }

    if corner_pts.rows() != 4 {
        panic!("Could not find puzzle");
    }

    let puzzle = four_point_transform(&img, &corner_pts);

    Some(puzzle)
}

fn extract_digit(cell: &Mat) -> Option<Mat> {
    let thresh = clear_border(cell);
    let mut pts = VectorOfVectorOfPoint::new();
    let ret = find_contours(&thresh, 
                  &mut pts, 
                  RETR_EXTERNAL, 
                  CHAIN_APPROX_SIMPLE, 
                  Point::new(0, 0));
    if ret.is_err() || pts.len() == 0 {
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

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

fn predict(model: &Model, cell: &Mat) -> u8 {
    let mut out = Mat::default();
    resize(&cell, &mut out, opencv::core::Size::new(28, 28), 0.0, 0.0, INTER_AREA)
        .expect("Could not resize image");

    let mut data = Vec::new();
    for i in 0..out.rows() {
        for j in 0..out.cols() {
            let pixel = out.at_2d::<u8>(i, j).expect("Could not get pixel");
            data.push(*pixel as f32 / 255.0);
        }
    }

    let input = tract_ndarray::arr1(&data).into_shape((1, 28, 28, 1))
        .expect("Could not reshape data")
        .into_tensor();

    let result = model.run(tvec![input.into()]).expect("Could not run model");

    let result = result[1].to_array_view::<f32>().expect("Could not get result");

    let pred_digit = result.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u8)
        .unwrap();

    return pred_digit;
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

    let mut img = imread("res/img/sudoku0.jpg", opencv::imgcodecs::IMREAD_GRAYSCALE)
        .expect("Could not read image");

    let deskewed = find_puzzle(&img).expect("Could not find puzzle");
    
    resize(&deskewed, 
           &mut img,
           Size::new(600, 600),
           0.0, 0.0,
           INTER_AREA)
        .expect("Could not resize image");

    let mut board = [[0; 9]; 9];

    let step_x = img.cols() / 9;
    let step_y = img.rows() / 9;

    let model = tensorflow()
        .model_for_path("res/model/frozen_graph.pb")
        .expect("Could not load model")
        .with_input_fact(0, f32::fact([1, 28, 28, 1]).into())
        .expect("Could not set input fact")
        .into_optimized()
        .expect("Could not optimize model")
        .into_runnable()
        .expect("Could not make model runnable");

    for y in 0..9 {
        for x in 0..9 {
            let start_x = x * step_x;
            let start_y = y * step_y;

            let mut cell = Mat::default();
            let roi = Rect::new(start_x, start_y, step_x, step_y);
            Mat::roi(&img, roi)
                .expect("Could not get ROI")
                .copy_to(&mut cell)
                .expect("Could not copy ROI to cell");

            let digit = extract_digit(&cell);
            let val = match digit {
                Some(digit) => predict(&model, &digit),
                None => 0,
            };

            board[y as usize][x as usize] = val;
        }
    }

    print_board(&board);

    imshow("Display window", &img).expect("Could not show image");
    wait_key(0).expect("Could not wait for key");
}
