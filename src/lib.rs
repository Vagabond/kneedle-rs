mod maths;
use approx_eq::assert_approx_eq;

use crate::maths::{gaussian_smooth2d, minmax_normalize};

fn find_candidate_indices(data: Vec<Vec<f64>>, find_minima: bool) -> Vec<usize> {
    let rows = data.len();
    let mut candidates: Vec<usize> = Vec::new();
    for i in 1..(rows - 1) {
        let prev = data[i - 1][1];
        let cur = data[i][1];
        let next = data[i + 1][1];
        if (find_minima && prev > cur && next > cur) || (!find_minima && prev < cur && next < cur) {
            candidates.push(i);
        }
    }
    candidates
}

#[allow(dead_code)]
fn find_elbow_index(data: &[f64]) -> usize {
    let mut best_score = f64::NAN;
    let mut best_index = 0;

    for i in 0..data.len() {
        if data[i].abs() > best_score {
            best_score = data[i];
            best_index = i;
        }
    }
    best_index
}

fn prepare<I: AsRef<[f64]>>(
    data: &[I],
    smoothing_window: usize,
) -> Result<Vec<Vec<f64>>, &'static str> {
    //smooth the data to make local minimum/maximum easier to find (this is Step 1 in the paper)
    let smoothed_data = gaussian_smooth2d(data, smoothing_window)?;

    //prepare the data into the unit range (step 2 of paper)
    let mut normalized_data = minmax_normalize(smoothed_data)?;

    //subtract normalised x from normalised y (this is step 3 in the paper)
    for i in 0..normalized_data.len() {
        normalized_data[i][1] -= normalized_data[i][0]
    }

    Ok(normalized_data)
}

fn compute_average_variance(data: Vec<Vec<f64>>) -> f64 {
    let mut variance = 0.0;

    for i in 0..data.len() - 1 {
        variance += data[i + 1][0] - data[i][0];
    }
    variance / (data.len() - 1) as f64
}


pub fn flip_x<I>(
    data: &[I],
    ) -> Vec<Vec<f64>> 
where
    I: AsRef<[f64]> + Clone,
{
    let mut xmax = f64::MIN;

    for i in 0..data.len() {
        xmax = xmax.max(data[i].as_ref()[0]);
    }

    let mut out = Vec::new();

    for row in data.iter().rev() {
        out.push(vec![xmax - row.as_ref()[0], row.as_ref()[1]]);
    }

    out
}

pub fn kneedle<I>(
    data: &[I],
    s: i32,
    smoothing_window: usize,
    find_elbow: bool,
) -> Result<Vec<I>, &'static str>
where
    I: AsRef<[f64]> + Clone,
{
    if data.is_empty() {
        return Err("Empty data");
    }

    let datasize = data.len();

    if data[0].as_ref().len() != 2 {
        return Err("all data should be 2 dimensional");
    }

    //do steps 1,2,3 of the paper in the prepare method
    let normalized_data = prepare(data, smoothing_window)?;

    //find candidate indices (this is step 4 in the paper)
    let candidate_indices = find_candidate_indices(normalized_data.clone(), find_elbow);

    //go through each candidate index, i, and see if the indices after i are satisfy the threshold requirement
    //(this is step 5 in the paper)

    let mut step = compute_average_variance(normalized_data.clone());

    if find_elbow {
        step *= s as f64;
    } else {
        step *= -s as f64;
    }

    let mut local_min_max_pts: Vec<I> = Vec::new();

    //check each candidate to see if it is a real elbow/knee
    //(this is step 6 in the paper)
    for i in 0..candidate_indices.len() {
        let candidate_index = candidate_indices[i];
        let mut end = datasize;
        if i + 1 < candidate_indices.len() {
            end = candidate_indices[i + 1];
        }

        let threshold = normalized_data[candidate_index][1] + step;

        for j in (candidate_index + 1)..end {
            if (find_elbow && normalized_data[j][1] > threshold)
                || (!find_elbow && normalized_data[j][1] < threshold)
            {
                local_min_max_pts.push(data[candidate_index].clone());
                break;
            }
        }
    }
    Ok(local_min_max_pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let test_data = [
            [0.0, 0.0],
            [0.1, 0.55],
            [0.2, 0.75],
            [0.35, 0.825],
            [0.45, 0.875],
            [0.55, 0.9],
            [0.675, 0.925],
            [0.775, 0.95],
            [0.875, 0.975],
            [1.0, 1.0],
        ];

        let knee_points = kneedle(&test_data, 1, 1, false).unwrap();

        assert_eq!(1, knee_points.len());
        assert_approx_eq!(0.2, knee_points[0][0]);
        assert_approx_eq!(0.75, knee_points[0][1]);
    }

    #[test]
    fn figure2() {
        /*let test_data = [
            [0.0, -5.0],
            [0.11111111, 0.26315789],
            [0.22222222, 1.89655172],
            [0.33333333, 2.69230769],
            [0.44444444, 3.16326531],
            [0.55555556, 3.47457627],
            [0.66666667, 3.69565217],
            [0.77777778, 3.86075949],
            [0.88888889, 3.98876404],
            [1.0,        4.09090909],
        ];*/

    let mut test_data = vec![vec![0.0, -5.0]];

    // Figure 2 depicts how Kneedle works for data points drawn
    // from the curve y = −1/x + 5 where x-values are between 0
    // and 1.
    for i in 1..11 {
        test_data.push(vec![i as f64 / 10.0, (-1.0/i as f64) + 5.0]);
    }


    println!("data {:?}", test_data);

    let smoothed_data = gaussian_smooth2d(&test_data, 1).unwrap();

        println!("smoothed {:?}", smoothed_data);

            let normalized_data = prepare(&test_data, 1).unwrap();

        println!("normalized {:?}", normalized_data);

        let knee_points = kneedle(&test_data, 1, 1, false).unwrap();
        assert_eq!(1, knee_points.len());
        assert_approx_eq!(0.2, knee_points[0][0]);
        assert_approx_eq!(4.5, knee_points[0][1]);
    }

    #[test]
    fn convex_increasing() {
        let test_data = [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 10.0],
            [6.0, 15.0],
            [7.0, 20.0],
            [8.0, 40.0],
            [9.0, 100.0],
        ];

        let knee_points = kneedle(&test_data, 1, 1, true).unwrap();
        assert_eq!(1, knee_points.len());
        assert_approx_eq!(7.0, knee_points[0][0]);
        assert_approx_eq!(20.0, knee_points[0][1]);
    }

    #[test]
    fn convex_decreasing() {
        let test_data = [
            [0.0, 100.0],
            [1.0, 40.0],
            [2.0, 20.0],
            [3.0, 15.0],
            [4.0, 10.0],
            [5.0, 5.0],
            [6.0, 4.0],
            [7.0, 3.0],
            [8.0, 2.0],
            [9.0, 1.0],
        ];

        let knee_points = kneedle(&flip_x(&test_data), 1, 1, true).unwrap();
        assert_eq!(1, knee_points.len());
        assert_approx_eq!(7.0, knee_points[0][0]);
        assert_approx_eq!(20.0, knee_points[0][1]);
    }

    #[test]
    fn concave_decreasing() {
        let test_data = [
            [0.0, 99.0],
            [1.0, 98.0],
            [2.0, 97.0],
            [3.0, 96.0],
            [4.0, 95.0],
            [5.0, 90.0],
            [6.0, 85.0],
            [7.0, 80.0],
            [8.0, 60.0],
            [9.0, 0.0],
        ];

        let knee_points = kneedle(&flip_x(&test_data), 1, 1, false).unwrap();
        assert_eq!(1, knee_points.len());
        assert_approx_eq!(2.0, knee_points[0][0]);
        assert_approx_eq!(80.0, knee_points[0][1]);
    }

    #[test]
    fn concave_increasing() {
        let test_data = [
            [0.0, 0.0],
            [1.0, 60.0],
            [2.0, 80.0],
            [3.0, 85.0],
            [4.0, 90.0],
            [5.0, 95.0],
            [6.0, 96.0],
            [7.0, 97.0],
            [8.0, 98.0],
            [9.0, 99.0],
        ];

        let knee_points = kneedle(&test_data, 1, 1, false).unwrap();
        assert_eq!(1, knee_points.len());
        assert_approx_eq!(2.0, knee_points[0][0]);
        assert_approx_eq!(80.0, knee_points[0][1]);
    }

    #[test]
    fn bumpy() {
        let test_data = [
            [0.0, 7305.0],
            [1.0, 6979.0],
            [2.0, 6666.6],
            [3.0, 6463.2],
            [4.0, 6326.5],
            [5.0, 6048.8],
            [6.0, 6032.8],
            [7.0, 5762.0],
            [8.0, 5742.8],
            [9.0, 5398.2],
            [10.0, 5256.8],
            [11.0, 5227.0],
            [12.0, 5001.7],
            [13.0, 4942.0],
            [15.0, 4854.2],
            [16.0, 4734.6],
            [17.0, 4558.7],
            [18.0, 4491.1],
            [19.0, 4411.6],
            [20.0, 4333.0],
            [21.0, 4234.6],
            [22.0, 4139.1],
            [23.0, 4056.8],
            [24.0, 4022.5],
            [25.0, 3868.0],
            [26.0, 3808.3],
            [27.0, 3745.3],
            [28.0, 3692.3],
            [29.0, 3645.6],
            [30.0, 3618.3],
            [31.0, 3574.3],
            [32.0, 3504.3],
            [33.0, 3452.4],
            [34.0, 3401.2],
            [35.0, 3382.4],
            [36.0, 3340.7],
            [37.0, 3301.1],
            [38.0, 3247.6],
            [39.0, 3190.3],
            [40.0, 3180.0],
            [41.0, 3154.2],
            [42.0, 3089.5],
            [43.0, 3045.6],
            [44.0, 2989.0],
            [45.0, 2993.6],
            [46.0, 2941.3],
            [47.0, 2875.6],
            [48.0, 2866.3],
            [49.0, 2834.1],
            [50.0, 2785.1],
            [51.0, 2759.7],
            [52.0, 2763.2],
            [53.0, 2720.1],
            [54.0, 2660.1],
            [55.0, 2690.2],
            [56.0, 2635.7],
            [57.0, 2632.9],
            [58.0, 2574.6],
            [59.0, 2556.0],
            [60.0, 2545.7],
            [61.0, 2513.4],
            [62.0, 2491.6],
            [63.0, 2496.0],
            [64.0, 2466.5],
            [65.0, 2442.7],
            [66.0, 2420.5],
            [67.0, 2381.5],
            [70.0, 2388.1],
            [71.0, 2340.6],
            [72.0, 2335.0],
            [73.0, 2318.9],
            [74.0, 2319.0],
            [75.0, 2308.2],
            [76.0, 2262.2],
            [77.0, 2235.8],
            [78.0, 2259.3],
            [79.0, 2221.0],
            [80.0, 2202.7],
            [81.0, 2184.3],
            [82.0, 2170.1],
            [83.0, 2160.0],
            [84.0, 2127.7],
            [85.0, 2134.7],
            [86.0, 2102.0],
            [87.0, 2101.4],
            [88.0, 2066.4],
            [89.0, 2074.3],
            [90.0, 2063.7],
            [91.0, 2048.1],
            [92.0, 2031.9],
            ];
        let knee_points = kneedle(&flip_x(&test_data), 1, 1, true).unwrap();
        assert_eq!(1, knee_points.len());
        //assert_approx_eq!(7.0, knee_points[0][0]);
        assert_approx_eq!(15.0, knee_points[0][1]);
    }

}
