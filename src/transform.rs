use opencv::types::VectorOfPoint2f;
use opencv::core::*;
use opencv::imgproc::*;

/// This function takes a point from each row and
/// sums the x and y values and returns them as a
/// vector of floats.
pub fn sum_rows(mat: &VectorOfPoint2f) -> Vec<f32>{
    let mut sum = vec![0.0; mat.len() as usize];
    for i in 0..mat.len() {
        let p = mat.get(i as usize).unwrap();
        sum[i as usize] = p.x + p.y;
    }
    sum
}

/// This function takes a point from each row and
/// subtracts the x and y values and returns them as a
/// vector of floats.
pub fn diff_rows(mat: &VectorOfPoint2f) -> Vec<f32>{
    let mut diff = vec![0.0; mat.len() as usize];
    for i in 0..mat.len() {
        let p = mat.get(i as usize).unwrap();
        diff[i as usize] = p.x - p.y;
    }
    diff
}

/// This function retuens the index of the minimum value
/// in a slice of floats.
pub fn min_index(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// This function retuens the index of the maximum value
/// in a slice of floats.
pub fn max_index(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

/// This function takes a vector of points and returns
/// a vector of points with the points sorted in a
/// clockwise order.
pub fn order_points(pts: &VectorOfPoint2f) -> VectorOfPoint2f {

    let mut rect = VectorOfPoint2f::new();
    rect.push(Point2f::new(0.0, 0.0));
    rect.push(Point2f::new(0.0, 0.0));
    rect.push(Point2f::new(0.0, 0.0));
    rect.push(Point2f::new(0.0, 0.0));

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

/// This function returns the distance between two points.
pub fn distance(p1: &Point2f, p2: &Point2f) -> f32 {
    let x = p1.x - p2.x;
    let y = p1.y - p2.y;
    (x * x + y * y).sqrt()
}

/// This function converts a matrix to a vector of points.
pub fn mat_to_vector_of_point(mat: &Mat) -> VectorOfPoint2f {
    let mut vec = VectorOfPoint2f::new();
    for i in 0..mat.rows() {
        let p = mat.at_2d::<Point>(i, 0).unwrap();
        vec.push(Point2f::new(p.x as f32, p.y as f32));
    }
    vec
}

/// This function applies a perspective transform to an image.
pub fn four_point_transform(image: &Mat, pts: &Mat) -> Mat {

    // converted to VectorOfPoint2f for easier manipulation
    // VectorOfPoint2f is a vector of CV_32FC2 points
    let pts = mat_to_vector_of_point(pts);

    let rect = order_points(&pts);

    let tl = rect.get(0).unwrap();
    let tr = rect.get(1).unwrap();
    let br = rect.get(2).unwrap();
    let bl = rect.get(3).unwrap();

    let width_a = distance(&br, &bl);
    let width_b = distance(&tr, &tl);
    let max_width = width_a.max(width_b) - 1.0;

    let height_a = distance(&tr, &br);
    let height_b = distance(&tl, &bl);
    let max_height = height_a.max(height_b) - 1.0;

    // construct the destination points which will be used to
    // map the screen to a top-down, "birds eye" view
    let dst = Mat::from_slice_2d(&[
        [0.0, 0.0],
        [max_width, 0.0],
        [max_width, max_height],
        [0.0, max_height],
    ]).unwrap();

    let m = get_perspective_transform(&rect, &dst, DECOMP_LU).unwrap();

    let mut warped = Mat::default();
    warp_perspective(&image, 
                     &mut warped, 
                     &m, 
                     Size::new(max_width as i32, max_height as i32), 
                     INTER_LINEAR, 
                     BORDER_CONSTANT, 
                     Scalar::all(0.0)).unwrap();

    let mut out_warped = Mat::default();
    transpose(&warped, &mut out_warped).unwrap();

    return out_warped;
}
