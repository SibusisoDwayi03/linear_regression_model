use burn::autodiff::ADBackendDecorator;
use burn::backend::Wgpu;
use burn::data::dataloader::DataLoader;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::gradients::GradientsParams;
use burn::module::Module;
use burn::optim::{Adam, Optimizer};
use burn::tensor::{backend::Backend, Data, Tensor};
use rand::Rng;
use textplots::{Chart, Plot, Shape};

type Backend = ADBackendDecorator<Wgpu>;

#[derive(Module, Debug)]
struct LinearRegression {
    weight: Tensor<Backend, 1>,
    bias: Tensor<Backend, 1>,
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            weight: Tensor::from_floats([0.0]), // Initialize weight to 0
            bias: Tensor::from_floats([0.0]),   // Initialize bias to 0
        }
    }

    fn forward(&self, x: Tensor<Backend, 1>) -> Tensor<Backend, 1> {
        self.weight.mul(x).add(self.bias)
    }
}

// Generate synthetic data
fn generate_data(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();

    for _ in 0..n {
        let x: f32 = rng.gen_range(-10.0..10.0);
        let y = 2.0 * x + 1.0 + rng.gen_range(-0.5..0.5); // Adding noise
        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}

fn main() {
    let model = LinearRegression::new();
    let optimizer = Adam::new(0.01); // Learning rate = 0.01
    let epochs = 100;

    let (x_train, y_train) = generate_data(100);
    let x_train_tensor = Tensor::<Backend, 1>::from_data(Data::from(x_train.clone()));
    let y_train_tensor = Tensor::<Backend, 1>::from_data(Data::from(y_train.clone()));

    for epoch in 0..epochs {
        let predictions = model.forward(x_train_tensor.clone());
        let loss = predictions.sub(y_train_tensor.clone()).powf(2.0).mean(); // MSE

        // Backpropagation and optimization step
        let gradients = loss.backward();
        model.weight = model.weight.clone() - optimizer.step(&gradients);
        model.bias = model.bias.clone() - optimizer.step(&gradients);

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.into_scalar());
        }
    }

    // Testing and visualization
    let (x_test, y_test) = generate_data(20);
    let x_test_tensor = Tensor::<Backend, 1>::from_data(Data::from(x_test.clone()));
    let y_pred_tensor = model.forward(x_test_tensor);

    let y_pred: Vec<f32> = y_pred_tensor.into_data().convert();
    println!("Actual vs Predicted:");
    Chart::new(120, 40, -10.0, 10.0)
        .lineplot(&Shape::Points(&x_test.iter().zip(y_test.iter()).map(|(&x, &y)| (x, y)).collect::<Vec<_>>()))
        .lineplot(&Shape::Points(&x_test.iter().zip(y_pred.iter()).map(|(&x, &y)| (x, y)).collect::<Vec<_>>()))
        .display();
}
