package File

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// StructData represents data and expected output for training
type StructData struct {
	Data           []float32
	ExpectedOutput []float32
}

// NewStructData is a function to initialize StructData with given data and expected output
func NewStructData(data []float32, expectedOutput []float32) *StructData {
	return &StructData{
		Data:           data,
		ExpectedOutput: expectedOutput,
	}
}

// ReadFile represents a structure that handles reading files
type ReadFile struct{}

// CreateTrainingData reads the file and creates training data
func (rf *ReadFile) CreateTrainingData(trainingDataset int, name string) ([]*StructData, error) {
	// Allocate a slice to hold the training data
	structOfData := make([]*StructData, trainingDataset)

	// Open the file for reading
	file, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("could not open file: %v", err)
	}
	defer file.Close()

	// Create a buffered reader
	reader := bufio.NewScanner(file)

	// Process each line
	i := 0
	for reader.Scan() {
		line := reader.Text()

		// Split the line by tabs
		currencies := strings.Split(line, "\t")

		// Parse input values
		input := make([]float32, 2)
		f0, _ := strconv.ParseFloat(currencies[0], 32) // ParseFloat always returns float64
		input[0] = float32(f0)                         // Explicitly convert to float32

		f1, _ := strconv.ParseFloat(currencies[1], 32) // Same for the second value
		input[1] = float32(f1)

		// Parse expected output based on the third value
		var expectedOutput []float32
		switch currencies[2] {
		case "C1":
			expectedOutput = []float32{1, 0, 0}
		case "C2":
			expectedOutput = []float32{0, 1, 0}
		case "C3":
			expectedOutput = []float32{0, 0, 1}
		default:
			// Handle unexpected cases (optional)
			return nil, fmt.Errorf("unexpected value in third column: %s", currencies[2])
		}

		// Create a new StructData and store it in the array
		structOfData[i] = NewStructData(input, expectedOutput)
		i++
	}

	// Check for reading errors (other than EOF)
	if err := reader.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	return structOfData, nil
}
