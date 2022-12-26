use opencv::{core::*, types::VectorOfPoint};
use opencv::imgproc::*;

// VectorOfPoint is a vector of CV_32SC2 points
// CV_32SC2 is a 2D point with signed integer coordinates
pub fn sum_rows(mat: &VectorOfPoint) -> Vec<i32>{
    let mut sum = vec![0; mat.len() as usize];
    for i in 0..mat.len() {
        let p = mat.get(i as usize).unwrap();
        sum[i as usize] = p.x + p.y;
    }
    sum
}

// VectorOfPoint is a vector of CV_32SC2 points
// CV_32SC2 is a 2D point with signed integer coordinates
pub fn diff_rows(mat: &VectorOfPoint) -> Vec<i32>{
    let mut diff = vec![0; mat.len() as usize];
    for i in 0..mat.len() {
        let p = mat.get(i as usize).unwrap();
        diff[i as usize] = p.x - p.y;
    }
    diff
}

pub fn min_index(arr: &[i32]) -> usize {
    arr.iter()
        .enumerate()
        .min_by_key(|&(_, v)| v)
        .map(|(i, _)| i)
        .unwrap()
}

pub fn max_index(arr: &[i32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by_key(|&(_, v)| v)
        .map(|(i, _)| i)
        .unwrap()
}

pub fn order_points(pts: &VectorOfPoint) -> VectorOfPoint {

    let mut rect = VectorOfPoint::new();
    rect.push(Point::new(0, 0));
    rect.push(Point::new(0, 0));
    rect.push(Point::new(0, 0));
    rect.push(Point::new(0, 0));


    let sum = sum_rows(pts);

    let tl = min_index(&sum);
    let br = max_index(&sum);
    rect.set(0, pts.get(tl).unwrap()).unwrap();
    rect.set(2, pts.get(br).unwrap()).unwrap();

    let diff = diff_rows(pts);

    let tr = min_index(&diff);
    let bl = max_index(&diff);
    rect.set(1, pts.get(tr).unwrap()).unwrap();
    rect.set(3, pts.get(bl).unwrap()).unwrap();

    rect
}

pub fn distance(p1: &Point, p2: &Point) -> f64 {
    let x = p1.x - p2.x;
    let y = p1.y - p2.y;
    ((x * x + y * y) as f64).sqrt()
}

pub fn mat_to_vector_of_point(mat: &Mat) -> VectorOfPoint {
    let mut vec = VectorOfPoint::new();
    for i in 0..mat.rows() {
        let p = mat.at_2d::<Point>(i, 0).unwrap();
        vec.push(Point::new(p.x, p.y));
    }
    vec
}

pub fn four_point_transform(image: &Mat, pts: &Mat) -> Mat {

    let pts = mat_to_vector_of_point(pts);

    let rect = order_points(&pts);

    let tl = rect.get(0).unwrap();
    let tr = rect.get(1).unwrap();
    let br = rect.get(2).unwrap();
    let bl = rect.get(3).unwrap();

    let width_a = distance(&br, &bl);
    let width_b = distance(&tr, &tl);
    let max_width = width_a.max(width_b);

    let height_a = distance(&tr, &br);
    let height_b = distance(&tl, &bl);
    let max_height = height_a.max(height_b);

    let mut dst = VectorOfPoint::new();
    dst.push(Point::new(0, 0));
    dst.push(Point::new(max_width as i32 - 1, 0));
    dst.push(Point::new(max_width as i32 - 1, max_height as i32 - 1));
    dst.push(Point::new(0, max_height as i32 - 1));

    let m = get_perspective_transform(&rect, &dst, DECOMP_LU).unwrap();

    let mut warped = Mat::default();
    warp_perspective(&image, 
                     &mut warped, 
                     &m, 
                     Size::new(max_width as i32, max_height as i32), 
                     INTER_LINEAR, 
                     BORDER_CONSTANT, 
                     Scalar::all(0.0)).unwrap();

    return warped;
}
