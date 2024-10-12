package main

import (
	"NeuralNetwork/Calculations"
	"NeuralNetwork/File"
	"NeuralNetwork/neuronika"
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

const (
	Threshold          = 0.001
	learningRate       = 0.01
	batchSize          = 50
	minIterations      = 700
	inputLayer         = 2
	firstHidden        = 2
	secondHidden       = 2
	thirdHidden        = 2
	outputLayer        = 3
	SizeOfData         = 4000
	numLayers          = 4
	activationFunction = 3   // 1 = sigmoid, 2 = relu, 3 = tanh
	bias               = 1.0 // Bias for activation functions
)

var (
	iteration int
	layers    [][]neuronika.Neuron
	TrainData []neuronika.StructData
	TestData  []neuronika.StructData
)

func printTrainData(trainData []neuronika.StructData) {
	for i, data := range trainData {
		fmt.Printf("TrainData[%d]:\n", i)
		fmt.Println("  Data:", data.Data)
		fmt.Println("  ExpectedOutput:", data.ExpectedOutput)
	}
}
func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	// Initialize neural network layers
	initializeNetwork()

	// Create and load training data
	File.CreateFile(SizeOfData, "trainData.txt")
	TrainData = loadDataFromFile("trainData.txt")

	// Train the neural network
	fmt.Println("Training started...")
	//	printTrainData(TrainData)
	Train(TrainData)

	fmt.Println("problem..")
	// Create and load test data
	File.CreateFile(SizeOfData, "testData.txt")
	TestData = loadDataFromFile("testData.txt")

	// Test the trained neural network
	fmt.Println("Testing started...")
	Test(TestData)
}

// Initialize neural network layers with random weights
func initializeNetwork() {
	layers = make([][]neuronika.Neuron, numLayers)
	layers[0] = InitializeLayer(inputLayer, firstHidden)   // Input to first hidden layer
	layers[1] = InitializeLayer(firstHidden, secondHidden) // First hidden to second hidden
	layers[2] = InitializeLayer(secondHidden, thirdHidden) // Second hidden to third hidden
	layers[3] = InitializeLayer(thirdHidden, outputLayer)  // Third hidden to output layer
	fmt.Println("Neural network initialized.")
}

// Train the neural network with the provided data
func Train(TrainData []neuronika.StructData) {
	prevSSE := math.MaxFloat64
	totalsize := len(TrainData) / batchSize

	for {
		var SSE float64
		offset := 0

		// Process data in batches
		for j := 0; j < totalsize-1; j++ {

			for k := 0; k < batchSize; k++ {

				initializeInputs(TrainData[k+offset].Data)

				forward()

				SSE += backward(TrainData[k+offset])

			}

			updateWeightsOfBatch(learningRate)
			offset += batchSize
		}
		fmt.Println("problem.2.")
		// Log the total SSE and stop if below threshold after minimum iterations
		fmt.Printf("Epoch %d - SSE: %.6f\n", iteration, SSE)
		if (SSE-prevSSE < Threshold) && iteration >= minIterations {
			fmt.Println("Training completed.")
			break
		}

		prevSSE = SSE
		iteration++
	}
}

// Test the neural network with the provided test data
func Test(TestData []neuronika.StructData) {
	// Open files for output
	c1, _ := os.Create("apotelesmataSostaC1.txt")
	c2, _ := os.Create("apotelesmataSostaC2.txt")
	c3, _ := os.Create("apotelesmataSostaC3.txt")
	writer2, _ := os.Create("apotelesmataLathos.txt")

	defer c1.Close()
	defer c2.Close()
	defer c3.Close()
	defer writer2.Close()

	// Track correct predictions
	correctCount := 0

	for i := 0; i < len(TestData); i++ {
		initializeInputs(TestData[i].Data)
		forward()

		// Identify the true and predicted class
		indexTrue, indexPredicted := getMaxIndex(TestData[i].ExpectedOutput), getMaxIndex(getOutputLayerValues())

		// Write results to files
		if indexTrue == indexPredicted {
			correctCount++
			writeResult(indexTrue, TestData[i], c1, c2, c3, "+")
		} else {
			writeResult(indexPredicted, TestData[i], writer2, nil, nil, "-")
		}
	}

	// Calculate and print accuracy
	accuracy := float64(correctCount) / float64(len(TestData)) * 100
	fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correctCount, len(TestData))
}

// Helper to load data from file and convert to struct format
func loadDataFromFile(filename string) []neuronika.StructData {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil
	}
	defer file.Close()

	var data []neuronika.StructData
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		var x1, x2 float32
		var classLabel string
		_, err := fmt.Sscanf(scanner.Text(), "%f\t%f\t%s", &x1, &x2, &classLabel)
		if err != nil {
			break
		}

		var expectedOutput []float32
		switch classLabel {
		case "C1":
			expectedOutput = []float32{1, 0, 0}
		case "C2":
			expectedOutput = []float32{0, 1, 0}
		case "C3":
			expectedOutput = []float32{0, 0, 1}
		}

		data = append(data, neuronika.StructData{
			Data:           []float32{x1, x2},
			ExpectedOutput: expectedOutput,
		})
	}

	return data
}

// Write the classification result to the appropriate file
func writeResult(index int, testData neuronika.StructData, correctC1, correctC2, correctC3 *os.File, result string) {
	switch index {
	case 0:
		if correctC1 != nil {
			writeToFile(correctC1, testData.Data, result)
		}
	case 1:
		if correctC2 != nil {
			writeToFile(correctC2, testData.Data, result)
		}
	case 2:
		if correctC3 != nil {
			writeToFile(correctC3, testData.Data, result)
		}
	}
}

func writeToFile(writer *os.File, data []float32, result string) {
	bufferedWriter := bufio.NewWriter(writer)
	bufferedWriter.WriteString(fmt.Sprintf("%.2f\t%.2f\t%s\n", data[0], data[1], result))
	bufferedWriter.Flush()
}

// Utility function to get max index from an array of float values
func getMaxIndex(arr []float32) int {
	maxIndex := 0
	for i := 1; i < len(arr); i++ {
		if arr[i] > arr[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}

// Get the output layer values (final layer neurons)
func getOutputLayerValues() []float32 {
	lastLayer := len(layers) - 1
	outputValues := make([]float32, len(layers[lastLayer]))
	for i := 0; i < len(layers[lastLayer]); i++ {
		outputValues[i] = layers[lastLayer][i].Value
	}
	return outputValues
}

func InitializeLayer(prevNeurons, numberNeurons int) []neuronika.Neuron {
	layer := make([]neuronika.Neuron, numberNeurons)
	for i := 0; i < numberNeurons; i++ {
		// Initialize weights and error slices based on previous layer size
		weights := make([]float32, prevNeurons)
		error := make([]float32, prevNeurons) // Error slice should also be based on previous neurons

		for j := 0; j < prevNeurons; j++ {
			weights[j] = rand.Float32()*2 - 1 // Random weights between -1 and 1
			error[j] = 0                      // Initialize error to 0
		}

		neuron := neuronika.Neuron{}
		neuron.SetterNeuron(weights, error) // Set weights and error slice
		layer[i] = neuron
	}
	return layer
}

func CreateTrainingData(filename string) []neuronika.StructData {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil
	}
	defer file.Close()

	var trainData []neuronika.StructData
	for {
		var x1, x2 float32
		var classLabel string
		_, err := fmt.Fscanf(file, "%f\t%f\t%s\n", &x1, &x2, &classLabel)
		if err != nil {
			break
		}

		var expectedOutput []float32
		switch classLabel {
		case "C1":
			expectedOutput = []float32{1, 0, 0}
		case "C2":
			expectedOutput = []float32{0, 1, 0}
		case "C3":
			expectedOutput = []float32{0, 0, 1}
		}

		data := neuronika.StructData{
			Data:           []float32{x1, x2},
			ExpectedOutput: expectedOutput,
		}
		trainData = append(trainData, data)
	}

	return trainData
}
func initializeInputs(inputs []float32) {
	layers[0] = make([]neuronika.Neuron, len(inputs))
	for i := 0; i < len(inputs); i++ {
		neuron := neuronika.Neuron{}
		neuron.NeuronForInput(inputs[i]) // Set the neuron as an input neuron
		layers[0][i] = neuron
	}
}

// func initializeInputs(inputs []float32) {
// 	// Ensure layers is properly initialized
// 	if layers[0] == nil {
// 		layers[0] = make([]neuronika.Neuron, len(inputs))
// 	} else if len(layers[0]) != len(inputs) {
// 		// Optionally resize the layer if it doesn't match
// 		layers[0] = make([]neuronika.Neuron, len(inputs))
// 	}

// 	// Initialize each neuron in the input layer
// 	for i := 0; i < len(inputs); i++ {
// 		neuron := neuronika.Neuron{}
// 		neuron.NeuronForInput(inputs[i]) // Set the neuron as input neuron
// 		layers[0][i] = neuron
// 	}
// }

func forward() {
	// Hidden layers
	for i := 1; i < len(layers)-1; i++ {
		for j := 0; j < len(layers[i]); j++ {
			sum := float32(0)
			for k := 0; k < len(layers[i-1]); k++ {
				sum += layers[i-1][k].Value * layers[i][j].Weights[k]
			}
			sum += bias

			// Choose activation function using your Calculations package
			switch activationFunction {
			case 1:
				layers[i][j].Value = Calculations.Sigmoid(sum)
			case 2:
				layers[i][j].Value = Calculations.ReLU(sum)
			case 3:
				layers[i][j].Value = Calculations.Tanh(sum)
			}
		}
	}

	// Output layer
	lastLayer := len(layers) - 1
	for j := 0; j < len(layers[lastLayer]); j++ {
		sum := float32(0)
		for k := 0; k < len(layers[lastLayer-1]); k++ {
			sum += layers[lastLayer-1][k].Value * layers[lastLayer][j].Weights[k]
		}
		sum += bias
		layers[lastLayer][j].Value = Calculations.Sigmoid(sum)
	}
}
func backward(tData neuronika.StructData) float64 {
	numberLayers := len(layers)

	outIndex := numberLayers - 1
	totalError := 0.0

	// Output layer
	outputNeurons := len(layers[outIndex])

	for i := 0; i < outputNeurons; i++ {
		// For each neuron in the output layer
		output := layers[outIndex][i].Value

		target := tData.ExpectedOutput[i]

		totalError += math.Pow(float64(target-output), 2)

		delta := (target - output) * Calculations.SigmoidDerivative(output)
		layers[outIndex][i].Gradient = delta

		// For each weight of the neuron

		for j := 0; j < len(layers[outIndex][i].Weights); j++ {

			previousOutput := layers[outIndex-1][j].Value
			error := -delta * previousOutput
			layers[outIndex][i].Error[j] += error
		}
	}

	// Hidden layers
	for i := outIndex - 1; i > 0; i-- {
		// For all neurons in that layer

		for j := 0; j < len(layers[i]); j++ {
			output := layers[i][j].Value

			gradientSum := 0.0
			for k := 0; k < len(layers[i+1]); k++ {
				curNeuron := layers[i+1][k]

				gradientSum += float64(curNeuron.Weights[j] * curNeuron.Gradient)
			}

			var delta float32
			switch activationFunction {
			case 1:
				delta = float32(gradientSum) * Calculations.SigmoidDerivative(output) // Sigmoid derivative
			case 2:
				delta = float32(gradientSum) * Calculations.ReLUDerivative(output) // ReLU derivative
			case 3:
				delta = float32(gradientSum) * Calculations.TanhDerivative(output) // Tanh derivative
			}

			layers[i][j].Gradient = delta

			// For all their weights
			for k := 0; k < len(layers[i][j].Weights); k++ {

				//fmt.Printf("Layer: %d, Neuron: %d, Weights size: %d, Error size: %d\n", i, j, len(layers[i][j].Weights), len(layers[i][j].Error))
				//	fmt.Println(len(layers[i-1]))
				previousOutput := layers[i-1][k].Value

				//	fmt.Printf("Layer: %d, Neuron: %d, Weights size: %d, Error size: %d\n", i, j, len(layers[i][j].Weights), len(layers[i][j].Error))
				layers[i][j].Error[k] += -delta * previousOutput
			}
		}
	}

	return totalError
}

func updateWeightsOfBatch(learningRate float32) {
	lastLayer := len(layers) - 1

	// Update output layer
	for i := 0; i < len(layers[lastLayer]); i++ {
		for j := 0; j < len(layers[lastLayer][i].Weights); j++ {
			layers[lastLayer][i].Weights[j] -= learningRate * layers[lastLayer][i].Error[j]
			layers[lastLayer][i].Error[j] = 0
		}
	}

	// Update hidden layers
	for i := lastLayer - 1; i > 0; i-- {
		for j := 0; j < len(layers[i]); j++ {
			for k := 0; k < len(layers[i][j].Weights); k++ {
				layers[i][j].Weights[k] -= learningRate * layers[i][j].Error[k]
				layers[i][j].Error[k] = 0
			}
		}
	}
}
