mod transform;

#[cfg(test)]
mod tests {

    use opencv::types::VectorOfPoint;
    use opencv::core::Point;
    use opencv::imgcodecs::*;

    use super::transform::*;

    #[test]
    fn test_sum_rows() {
        let input = VectorOfPoint::from_iter(vec![
            Point::new(1, 2),
            Point::new(3, 4),
            Point::new(5, 6),
            Point::new(7, 8),
        ]);
        assert_eq!(sum_rows(&input), vec![3, 7, 11, 15]);
    }

    #[test]
    fn test_diff_rows() {
        let input = VectorOfPoint::from_iter(vec![
            Point::new(1, 2),
            Point::new(3, 4),
            Point::new(5, 6),
            Point::new(7, 8),
        ]);
        assert_eq!(diff_rows(&input), vec![-1, -1, -1, -1]);
    }

    #[test]
    fn test_min_index() {
        let input = vec![3, 7, 11, 15];
        assert_eq!(min_index(&input), 0);
    }

    #[test]
    fn test_max_index() {
        let input = vec![3, 7, 11, 15];
        assert_eq!(max_index(&input), 3);
    }

    #[test]
    fn test_order_points() {
        let input = VectorOfPoint::from_iter(vec![
            Point::new(0, 0),
            Point::new(800, 0),
            Point::new(0, 800),
            Point::new(800, 800),
        ]);
        let expected = VectorOfPoint::from_iter(vec![
            Point::new(0, 0),
            Point::new(0, 800),
            Point::new(800, 800),
            Point::new(800, 0),
        ]);
        assert_eq!(equal(&order_points(&input), &expected), true);
    }

    #[test]
    fn test_distance() {
        let p1 = Point::new(0, 0);
        let p2 = Point::new(800, 800);
        assert_eq!(distance(&p1, &p2) as u32, 1131);
    }

    fn equal(a: &VectorOfPoint, b: &VectorOfPoint) -> bool {
        let a = a.to_vec();
        let b = b.to_vec();
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(a, b)| a == b)
    }
}
