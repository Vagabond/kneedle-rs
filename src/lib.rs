mod maths;

use crate::maths::{gaussian_smooth2d, minmax_normalize};

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

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
    return candidates;
}

fn find_elbow_index(data: Vec<f64>) -> usize {
    let mut best_score = f64::NAN;
    let mut best_index = 0;

    for i in 0..data.len() {
        if data[i].abs() > best_score {
            best_score = data[i];
            best_index = i;
        }
    }
    return best_index;
}

fn prepare(data: Vec<Vec<f64>>, smoothing_window: usize) -> Result<Vec<Vec<f64>>, &'static str> {
    //smooth the data to make local minimum/maximum easier to find (this is Step 1 in the paper)
    let smoothed_data = gaussian_smooth2d(data, smoothing_window)?;

    //prepare the data into the unit range (step 2 of paper)
    let mut normalized_data = minmax_normalize(smoothed_data)?;

    //subtract normalised x from normalised y (this is step 3 in the paper)
    for i in 0..normalized_data.len() {
        normalized_data[i][1] = normalized_data[i][1] - normalized_data[i][0]
    }

    return Ok(normalized_data);
}

fn compute_average_variance(data: Vec<Vec<f64>>) -> f64 {
    let mut variance = 0.0;

    for i in 0..data.len() - 1 {
        variance += data[i + 1][0] - data[i][0];
    }
    return variance / (data.len() - 1) as f64;
}

pub fn kneedle(
    data: Vec<Vec<f64>>,
    s: i32,
    smoothing_window: usize,
    find_elbow: bool,
) -> Result<Vec<Vec<f64>>, &'static str> {
    if data.len() == 0 {
        return Err("Empty data");
    }

    let datasize = data.len();

    if data[0].len() != 2 {
        return Err("all data should be 2 dimensional");
    }

    //do steps 1,2,3 of the paper in the prepare method
    let normalized_data = prepare(data.clone(), smoothing_window)?;

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

    let mut local_min_max_pts: Vec<Vec<f64>> = Vec::new();

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
    return Ok(local_min_max_pts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
