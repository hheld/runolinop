#[allow(dead_code)]
pub fn norm2(v: &[f64]) -> f64 {
    norm2_sqr(v).sqrt()
}

#[allow(dead_code)]
pub fn norm2_sqr(v: &[f64]) -> f64 {
    v.iter().fold(0., |sum, v_i| sum + v_i * v_i)
}

#[allow(dead_code)]
pub fn inner_product(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() == b.len() {
        Some(
            a.iter()
                .zip(b.iter())
                .fold(0., |sum, a_b| sum + a_b.0 * a_b.1),
        )
    } else {
        None
    }
}

#[allow(dead_code)]
pub fn scaled(v: &[f64], s: f64) -> Vec<f64> {
    let mut scaled = Vec::from(v);

    for iter in scaled.iter_mut() {
        *iter *= s;
    }

    scaled
}

#[allow(dead_code)]
pub fn add(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    if a.len() == b.len() {
        let mut sum = Vec::from(a);

        for (iter_a, iter_b) in sum.iter_mut().zip(b.iter()) {
            *iter_a += iter_b;
        }

        Some(sum)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm2_test() {
        assert_eq!(norm2(&[4., 3.]), 5.);
    }

    #[test]
    fn norm2_sqr_test() {
        assert_eq!(norm2_sqr(&[4., 3.]), 25.);
    }

    #[test]
    fn inner_product_test() {
        assert_eq!(inner_product(&[1., 2.], &[3., 4.]).unwrap(), 11.);
        assert_eq!(inner_product(&[1., 2.], &[3., 4., 5.]), None);
    }

    #[test]
    fn scaled_test() {
        let v = vec![1., 2., 0., 3.];

        assert_eq!(scaled(&v, 2.), [2., 4., 0., 6.]);
    }

    #[test]
    fn add_test() {
        let a = vec![1., 0., 2.];
        let b = vec![3., 1., 1.];

        assert_eq!(add(&a, &b).unwrap(), [4., 1., 3.]);
        assert_eq!(add(&[1., 2.], &[1., 2., 3.]), None);
    }
}
