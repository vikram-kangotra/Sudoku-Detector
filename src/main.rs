mod transform;

use opencv::core::*;
use opencv::imgcodecs::imread;
use opencv::imgproc::*;
use opencv::highgui::{imshow, wait_key};

use crate::transform::four_point_transform;

fn main() {

    let mut img = imread("res/sudoku.webp", opencv::imgcodecs::IMREAD_COLOR)
        .expect("Could not read image");

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
                  opencv::core::Point::new(0, 0))
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

    imshow("Display window", &warped).expect("Could not show image");
    wait_key(0).expect("Could not wait for key");
}
