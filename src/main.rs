use std::fs::File;
use std::error::Error;

use csv::ReaderBuilder;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

const L1_SIZE: usize = 10;
const L2_SIZE: usize = 10;

struct Network {
    w_1: Array2<f32>,
    w_2: Array2<f32>,
    b_1: Array1<f32>,
    b_2: Array1<f32>,

    training_inter: TrainingInter,
}

#[derive(Default)]
struct TrainingInter {
    z_1: Array2<f32>,
    z_2: Array2<f32>,
    a_1: Array2<f32>,
    a_2: Array2<f32>,

    dw_1: Array2<f32>,
    dw_2: Array2<f32>,
    db_1: Array1<f32>,
    db_2: Array1<f32>,
}

impl Network {
    fn new(input_size: usize) -> Self {
        Self {
            w_1: Array::random((L1_SIZE, input_size), Uniform::new(-0.5, 0.5).unwrap()),
            b_1: Array::random(L1_SIZE, Uniform::new(-0.5, 0.5).unwrap()),
            w_2: Array::random((L2_SIZE, L1_SIZE), Uniform::new(-0.5, 0.5).unwrap()),
            b_2: Array::random(L2_SIZE, Uniform::new(-0.5, 0.5).unwrap()),

            training_inter: TrainingInter::default()
        }
    }

    fn forward_prop(&mut self, a_0: &Array2<f32>) {
        // println!("=== b1 ===\n{:?}\n", self.b_1);
        // println!("=== b1_T ===\n{:?}\n", self.b_1.to_shape((10,1)).unwrap());
        self.training_inter.z_1 = self.w_1.dot(a_0) + &self.b_1.to_shape((self.b_1.len() ,1)).unwrap();
        self.training_inter.a_1 = self.training_inter.z_1.map(|x| relu(*x));
        self.training_inter.z_2 = self.w_2.dot(&self.training_inter.a_1) + &self.b_2.to_shape((self.b_2.len(), 1)).unwrap();
        self.training_inter.a_2 = softmax(self.training_inter.z_2.clone());
        // println!("=== A2 ===\n{:?}\n", self.training_inter.a_2);
    }

    fn back_prop(&mut self, x: &Array2<f32>, y: &Array2<f32>) {
        let rep_m = 1.0 / x.dim().1 as f32;

        let dz_2 = &self.training_inter.a_2 - y;
        // println!("=== dZ2 ===\n{:?}", dz_2);
        self.training_inter.dw_2 = (rep_m * &dz_2).dot(&self.training_inter.a_1.t());
        self.training_inter.db_2 = rep_m * dz_2.sum_axis(Axis(1));

        let dz_1 = self.w_2.t().dot(&dz_2) * self.training_inter.z_1.map(|e| diff_relu(*e));
        self.training_inter.dw_1 = (rep_m * &dz_1).dot(&x.t());
        self.training_inter.db_1 = rep_m * dz_1.sum_axis(Axis(1));
    }

    fn update_params(&mut self, alpha: f32) {
        self.w_1 = &self.w_1 - alpha * &self.training_inter.dw_1;
        self.b_1 = &self.b_1 - alpha * &self.training_inter.db_1;
        self.w_2 = &self.w_2 - alpha * &self.training_inter.dw_2;
        self.b_2 = &self.b_2 - alpha * &self.training_inter.db_2;
    }

    fn gradient_descent(&mut self, x: &Array2<f32>, labels: Array1<f32>, iterations: u32, alpha: f32) {
        let y = one_hot(&labels);
        for i in 0..iterations {
            self.forward_prop(x);
            self.back_prop(x, &y);
            self.update_params(alpha);

            if i % 100 == 0 {
                println!("Iteration: {}", i);
                println!("=== Predictions ===\n{:?}", self.get_predictions());
                // println!("=== solutions ===\n{:?}", labels);
                println!("Accuracy: {}%\n", self.get_accuracy(&labels) * 100.0);
            }
        }
    }

    fn get_predictions(&self) -> Array1<f32> {
        let mut pred = Array1::zeros(self.training_inter.a_2.dim().1);

        for i in 0..self.training_inter.a_2.dim().1 {
            pred[i] = self.training_inter.a_2.column(i)
                .into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as f32)
                .unwrap().to_owned();
        };

        pred
    }

    fn get_accuracy(&self, solutions: &Array1<f32>) -> f32 {
        let preds = self.get_predictions();
        let mut count = 0;
        for i in 0..solutions.len() {
            if solutions[i] == preds[i] {
                count += 1;
            }
        }

        count as f32 / solutions.len() as f32
    }
}

fn main() {
    let data = read_csv("res/mnist_train.csv").unwrap();
    let labels = data.row(0).to_owned();
    let x = data.slice(s![1.., ..]).to_owned();
    let (m, _) = x.dim();

    let mut net = Network::new(m);
    net.gradient_descent(&x, labels, 1000, 0.1);
}

fn read_csv(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);

    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;

    for result in rdr.records().take(100) {
        let record = result?;
        cols = cols.max(record.len());

        for item in record.iter() {
            data.push(item.parse::<f32>()?);
        }

        rows += 1;
    }

    Ok(Array2::from_shape_vec((rows, cols), data)?.reversed_axes())
}

fn one_hot(x: &Array1<f32>) -> Array2<f32> {
    let mut y = Array2::zeros((10, x.len()));
    for col in 0..x.len() {
        y[(x[col] as usize,col)] = 1.0;
    }

    y
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
} 

fn diff_relu(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
} 

fn softmax(mut z: Array2<f32>) -> Array2<f32> {
    for mut col in z.axis_iter_mut(Axis(1)) {
        let max = col.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap().to_owned();
        col.map_inplace(|x| *x = (*x - max).exp());
        let sum = col.sum();
        col.map_inplace(|x| *x /= sum);
    }

    z
}
