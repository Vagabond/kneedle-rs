fn gaussian(x: f64, height: f64, center: f64, width: f64) -> f64 {
    height * (-(x - center) * (x - center) / (2.0 * width * width)).exp()
}

pub fn gaussian_smooth2d<I: AsRef<[f64]>>(
    data: &[I],
    w: usize,
) -> Result<Vec<Vec<f64>>, &'static str> {
    let datasize = data.len();
    if datasize == 0 {
        return Err("Empty data");
    }

    let dimensions = data[0].as_ref().len();

    if dimensions == 0 {
        return Err("dimension cannot be 0");
    }

    let mut smoothed: Vec<Vec<f64>> = vec![vec![0.0; dimensions]; datasize];

    for i in 0..datasize {
        if data[i].as_ref().len() != dimensions {
            return Err("all rows must have the same dimension");
        }

        let mut start = 0;
        let mut end = i + w;

        if 0 < i as i32 - w as i32 {
            start = i - w;
        }

        if datasize - 1 < i + w {
            end = datasize - 1;
        }

        let mut sum_weights: Vec<f64> = vec![0.0; dimensions];
        let mut sum_index_weight = 0.0;

        for j in start..end + 1 {
            let index_score = (((j as i32 - i as i32) / w as i32) as f64).abs();
            let index_weight = gaussian(index_score, 1.0, 0.0, 1.0);

            for n in 0..dimensions {
                sum_weights[n] += index_weight * data[j].as_ref()[n];
            }
            sum_index_weight += index_weight;
        }

        for n in 0..dimensions {
            smoothed[i][n] = sum_weights[n] / sum_index_weight;
        }
    }

    Ok(smoothed)
}

pub fn minmax_normalize(data: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, &'static str> {
    let datasize = data.len();
    if datasize == 0 {
        return Err("Empty data");
    }

    let dimensions = data[0].len();

    if dimensions == 0 {
        return Err("dimension cannot be 0");
    }

    let mut min_each_dimension: Vec<f64> = vec![0.0; dimensions];
    let mut max_each_dimension: Vec<f64> = vec![0.0; dimensions];

    for i in 0..dimensions {
        min_each_dimension[i] = f64::MAX;
        min_each_dimension[i] = f64::MIN_POSITIVE;
    }

    //1) get min and max for each dimension of the data
    for i in 0..datasize {
        if data[i].len() != dimensions {
            return Err("all rows must have the same dimension");
        }
        for d in 0..dimensions {
            min_each_dimension[d] = min_each_dimension[d].min(data[i][d]);
            max_each_dimension[d] = max_each_dimension[d].max(data[i][d]);
        }
    }

    //2) normalise the data using the min and max
    let mut range_each_dimension: Vec<f64> = vec![0.0; dimensions];
    for d in 0..dimensions {
        range_each_dimension[d] = max_each_dimension[d] - min_each_dimension[d];
    }

    let mut normalized: Vec<Vec<f64>> = vec![vec![0.0; dimensions]; datasize];

    for i in 0..datasize {
        for n in 0..dimensions {
            normalized[i][n] = (data[i][n] - min_each_dimension[n]) / range_each_dimension[n]
        }
    }

    Ok(normalized)
}
