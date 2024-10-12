package File

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

// RandomFloat generates a random float between a min and max value
func RandomFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

// CreateFile generates the dataset with the given trainingDataset size and writes it to the specified file
func CreateFile(trainingDataset int, fileName string) {
	// Create the file
	file, err := os.Create(fileName)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	// Write data to the file
	for i := 0; i < trainingDataset; i++ {
		x1 := RandomFloat(-1, 1)
		x2 := RandomFloat(-1, 1)

		// Write x1 and x2 to file
		_, err := file.WriteString(fmt.Sprintf("%.6f\t%.6f", x1, x2))
		if err != nil {
			fmt.Println("Error writing to file:", err)
			return
		}

		// Calculate class based on conditions
		var classLabel string
		if (math.Pow(x1-0.5, 2.0)+math.Pow(x2-0.5, 2.0) < 0.2) && x2 > 0.5 {
			classLabel = "C1"
		} else if (math.Pow(x1-0.5, 2.0)+math.Pow(x2-0.5, 2.0) < 0.2) && x2 < 0.5 {
			classLabel = "C2"
		} else if (math.Pow(x1+0.5, 2.0)+math.Pow(x2+0.5, 2.0) < 0.2) && x2 > -0.5 {
			classLabel = "C1"
		} else if (math.Pow(x1+0.5, 2.0)+math.Pow(x2+0.5, 2.0) < 0.2) && x2 < -0.5 {
			classLabel = "C2"
		} else if (math.Pow(x1-0.5, 2.0)+math.Pow(x2+0.5, 2.0) < 0.2) && x2 > -0.5 {
			classLabel = "C1"
		} else if (math.Pow(x1-0.5, 2.0)+math.Pow(x2+0.5, 2.0) < 0.2) && x2 < -0.5 {
			classLabel = "C2"
		} else if (math.Pow(x1+0.5, 2.0)+math.Pow(x2-0.5, 2.0) < 0.2) && x2 > 0.5 {
			classLabel = "C1"
		} else if (math.Pow(x1+0.5, 2.0)+math.Pow(x2-0.5, 2.0) < 0.2) && x2 < 0.5 {
			classLabel = "C2"
		} else {
			classLabel = "C3"
		}

		// Write class label to file
		_, err = file.WriteString(fmt.Sprintf("\t%s\n", classLabel))
		if err != nil {
			fmt.Println("Error writing to file:", err)
			return
		}
	}

	fmt.Println("File created successfully!")
}
