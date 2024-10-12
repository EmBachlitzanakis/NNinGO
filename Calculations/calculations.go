package Calculations

import (
	"math"
	"math/rand"
)

// RandomFloat generates a random float between min and max, with the possibility of negating it
func RandomFloat(min, max float32) float32 {
	a := rand.Float32() // Generates a float32 between 0.0 and 1.0
	num := min + rand.Float32()*(max-min)
	if a < 0.5 {
		return num
	}
	return -num
}

// Sigmoid applies the sigmoid activation function
func Sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

// ReLU applies the ReLU activation function
func ReLU(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// Tanh applies the tanh activation function
func Tanh(x float32) float32 {
	return float32((math.Exp(float64(x)) - math.Exp(-float64(x))) / (math.Exp(float64(x)) + math.Exp(-float64(x))))
}

// TanhDerivative calculates the derivative of the tanh function
func TanhDerivative(output float32) float32 {
	return 1 - (output * output)
}

// SigmoidDerivative calculates the derivative of the sigmoid function
func SigmoidDerivative(output float32) float32 {
	return output * (1 - output)
}

// ReLUDerivative calculates the derivative of the ReLU function
func ReLUDerivative(output float32) float32 {
	if output > 0 {
		return 1.0
	}
	return 0.0
}
