package neuronika

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

type StructData struct {
	data           []float32
	expectedOutput []float32
}

type Neuron struct {
	value    float32
	weights  []float32
	error    []float32
	gradient float32
}

func (n *Neuron) NeuronForInput(value float32) {
	n.value = value
}

func (n *Neuron) SetterNeuron(weights, error []float32) {
	n.weights = weights
	n.error = error
}

// Activation Functions
func Sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func SigmoidDerivative(x float32) float32 {
	return x * (1 - x)
}

func ReLU(x float32) float32 {
	if x < 0 {
		return 0
	}
	return x
}

func ReLUDerivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

func Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func TanhDerivative(x float32) float32 {
	return 1 - x*x
}

func RandomFloat(min, max float32) float32 {
	return min + rand.Float32()*(max-min)
}

type ReadFile struct{}

func NewReadFile() *ReadFile {
	return &ReadFile{}
}

// Placeholder for CreateTrainingData
func (rf *ReadFile) CreateTrainingData(size int, filename string) []StructData {
	// Your implementation for creating training data goes here
	return make([]StructData, size)
}

type CreateFile struct{}

func NewCreateFile() *CreateFile {
	return &CreateFile{}
}

// Placeholder for CreateFile
func (cf *CreateFile) CreateFile(size int, filename string) {
	// Your implementation for creating a file goes here
}

type NeuralNetwork struct {
	layers             [][]*Neuron
	TrainData          []StructData
	TestData           []StructData
	learningRate       float32
	threshold          float64
	batchSize          int
	activationFunction int
	bias               float32
	inputLayer         int
	firstHidden        int
	secondHidden       int
	thirdHidden        int
	outputLayer        int
	sizeOfData         int
	numberOfLayers     int
}

func NewNeuralNetwork() *NeuralNetwork {
	return &NeuralNetwork{
		learningRate:       0.02, // For tanh
		threshold:          0.101,
		batchSize:          40,
		activationFunction: 3, // 1 = sigmoid, 2 = relu, 3 = tanh
		bias:               1,
		inputLayer:         2,
		firstHidden:        30,
		secondHidden:       25,
		thirdHidden:        12,
		outputLayer:        3,
		sizeOfData:         4000,
		numberOfLayers:     5, // 5 if we have 3 hidden layers
	}
}

func (nn *NeuralNetwork) InitializeLayers() {
	nn.layers = make([][]*Neuron, nn.numberOfLayers)
	nn.layers[1] = nn.InitializeLayer(nn.inputLayer, nn.firstHidden)   // Hidden Layer
	nn.layers[2] = nn.InitializeLayer(nn.firstHidden, nn.secondHidden) // Hidden Layer
	nn.layers[3] = nn.InitializeLayer(nn.secondHidden, nn.thirdHidden) // Hidden Layer
	nn.layers[4] = nn.InitializeLayer(nn.thirdHidden, nn.outputLayer)  // Output Layer
}

func (nn *NeuralNetwork) Train() {
	nn.InitializeLayers()
	provideData := NewReadFile()
	newfile := NewCreateFile()

	newfile.CreateFile(nn.sizeOfData, "trainData.txt")
	nn.TrainData = provideData.CreateTrainingData(nn.sizeOfData, "trainData.txt")

	// Training loop
	PrevSSE := math.MaxFloat64
	SSE := 0.0
	totalSize := nn.sizeOfData / nn.batchSize

	for iteration := 0; ; iteration++ {
		offset := 0
		for j := 0; j < totalSize; j++ {
			for k := 0; k < nn.batchSize; k++ {
				nn.InitializeInputs(nn.TrainData[k+offset].data)
				nn.Forward()
				SSE += nn.Backward(nn.TrainData[k+offset])
			}
			nn.UpdateWeightsOfBatch(nn.learningRate)
			offset += nn.batchSize
		}

		// Stop if the difference is lesser than threshold
		if (SSE-PrevSSE < nn.threshold) && iteration >= 700 {
			fmt.Printf("epoch: %d\n", iteration)
			break
		}
		fmt.Printf("Total squared error of the epoch: %f\n", SSE)
		PrevSSE = SSE
		SSE = 0
	}

	// Create Test sample
	newfile.CreateFile(nn.sizeOfData, "testData.txt")
	nn.TestData = provideData.CreateTrainingData(nn.sizeOfData, "testData.txt")
	nn.Evaluate()
}

func (nn *NeuralNetwork) Evaluate() {
	c1, _ := os.Create("apotelesmataSostaC1.txt")
	c2, _ := os.Create("apotelesmataSostaC2.txt")
	c3, _ := os.Create("apotelesmataSostaC3.txt")
	writer2, _ := os.Create("apotelesmataLathos.txt")
	defer c1.Close()
	defer c2.Close()
	defer c3.Close()
	defer writer2.Close()

	count := 0
	for i := 0; i < len(nn.TrainData); i++ {
		nn.InitializeInputs(nn.TestData[i].data)
		nn.Forward()

		var maxTrue, maxCalc float32
		var indexTrue, indexCalc int

		for j := 0; j < 3; j++ {
			if maxTrue < nn.TestData[i].expectedOutput[j] {
				maxTrue = nn.TestData[i].expectedOutput[j]
				indexTrue = j
			}
			if maxCalc < nn.layers[nn.numberOfLayers-1][j].value {
				maxCalc = nn.layers[nn.numberOfLayers-1][j].value
				indexCalc = j
			}
		}

		if indexTrue == indexCalc {
			count++
			if indexTrue == 1 {
				fmt.Fprintf(c1, "%f\t%f\t+\n", nn.TestData[i].data[0], nn.TestData[i].data[1])
			} else if indexTrue == 2 {
				fmt.Fprintf(c2, "%f\t%f\t+\n", nn.TestData[i].data[0], nn.TestData[i].data[1])
			} else {
				fmt.Fprintf(c3, "%f\t%f\t+\n", nn.TestData[i].data[0], nn.TestData[i].data[1])
			}
		} else {
			fmt.Fprintf(writer2, "%f\t%f\t-\n", nn.TestData[i].data[0], nn.TestData[i].data[1])
		}
	}

	percentage := float64(count) / float64(nn.sizeOfData)
	fmt.Printf("Amount of the correct positioned: %d\n", count)
	fmt.Printf("Percentage of accuracy: %f %%\n", percentage*100)
}

func (nn *NeuralNetwork) InitializeInputs(inputs []float32) {
	nn.layers[0] = make([]*Neuron, len(inputs))
	for i := 0; i < len(inputs); i++ {
		nn.layers[0][i] = &Neuron{}
		nn.layers[0][i].NeuronForInput(inputs[i])
	}
}

func (nn *NeuralNetwork) Forward() {
	// Hidden layers
	for i := 1; i < len(nn.layers)-1; i++ {
		for j := 0; j < len(nn.layers[i]); j++ {
			sum := float32(0)
			for k := 0; k < len(nn.layers[i-1]); k++ {
				sum += nn.layers[i-1][k].value * nn.layers[i][j].weights[k]
			}
			sum += nn.bias // Add in the bias
			switch nn.activationFunction {
			case 1:
				nn.layers[i][j].value = Sigmoid(sum) // Sigmoid activation
			case 2:
				nn.layers[i][j].value = ReLU(sum) // ReLU activation
			case 3:
				nn.layers[i][j].value = Tanh(sum) // Tanh activation
			}
		}
	}

	// Output layer
	for i := len(nn.layers) - 1; i < len(nn.layers); i++ {
		for j := 0; j < len(nn.layers[i]); j++ {
			sum := float32(0)
			for k := 0; k < len(nn.layers[i-1]); k++ {
				sum += nn.layers[i-1][k].value * nn.layers[i][j].weights[k]
			}
			sum += nn.bias
			nn.layers[i][j].value = Sigmoid(sum)
		}
	}
}

func (nn *NeuralNetwork) Backward(tData StructData) float64 {
	numberLayers := len(nn.layers)
	outIndex := numberLayers - 1
	totalError := 0.0

	// Output layers
	outputNeurons := len(nn.layers[outIndex])
	for i := 0; i < outputNeurons; i++ {
		output := nn.layers[outIndex][i].value
		target := tData.expectedOutput[i]

		// Calculate cross-entropy loss
		if target == 1 {
			totalError -= float64(math.Log(float64(output))) // When target is 1
		} else {
			totalError -= float64(math.Log(float64(1 - output))) // When target is 0
		}

		// Calculate delta
		delta := (target - output) * SigmoidDerivative(output)
		nn.layers[outIndex][i].gradient = delta

		// Update the error for backpropagation
		for j := 0; j < len(nn.layers[outIndex][i].weights); j++ {
			previousOutput := nn.layers[outIndex-1][j].value
			error := -delta * previousOutput
			nn.layers[outIndex][i].error[j] += error
		}
	}

	// All hidden layers
	for i := outIndex - 1; i > 0; i-- {
		for j := 0; j < len(nn.layers[i]); j++ {
			output := nn.layers[i][j].value

			gradientSum := float32(0)
			for k := 0; k < len(nn.layers[i+1]); k++ {
				curNeuron := nn.layers[i+1][k]
				gradientSum += curNeuron.weights[j] * curNeuron.gradient
			}

			delta := float32(0)
			switch nn.activationFunction {
			case 1:
				delta = gradientSum * SigmoidDerivative(output)
			case 2:
				delta = gradientSum * ReLUDerivative(output)
			case 3:
				delta = gradientSum * TanhDerivative(output)
			}

			nn.layers[i][j].gradient = delta
			for k := 0; k < len(nn.layers[i][j].weights); k++ {
				previousOutput := nn.layers[i-1][k].value
				error := -delta * previousOutput
				nn.layers[i][j].error[k] += error
			}
		}
	}

	return totalError
}

func (nn *NeuralNetwork) UpdateWeightsOfBatch(learningRate float32) {
	for i := 1; i < len(nn.layers); i++ {
		for j := 0; j < len(nn.layers[i]); j++ {
			for k := 0; k < len(nn.layers[i][j].weights); k++ {
				nn.layers[i][j].weights[k] -= learningRate * nn.layers[i][j].error[k]
			}
		}
	}
}

func (nn *NeuralNetwork) InitializeLayer(inputSize, outputSize int) []*Neuron {
	layer := make([]*Neuron, outputSize)
	for i := 0; i < outputSize; i++ {
		neuron := &Neuron{
			weights: make([]float32, inputSize),
		}
		// Initialize weights with random values
		for j := 0; j < inputSize; j++ {
			neuron.weights[j] = RandomFloat(-1.0, 1.0)
		}
		layer[i] = neuron
	}
	return layer
}
