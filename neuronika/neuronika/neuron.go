package neuronika

// Neuron struct represents a neuron in a neural network
type Neuron struct {
	Weights  []float32
	Gradient float32
	Value    float32
	Error    []float32
}

// SetterNeuron initializes the neuron with weights and error array
func (n *Neuron) SetterNeuron(weights []float32, error []float32) {
	n.Weights = weights
	n.Error = error
}

// NeuronForInput initializes a neuron as an input neuron
// where weights are not needed and only the input value is set.
func (n *Neuron) NeuronForInput(value float32) {
	n.Weights = nil
	n.Value = value
}

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
