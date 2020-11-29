using System;
using System.Linq;
using System.Text.Json.Serialization;

namespace neuralnet1
{

    public class Neuron
    {
        public double Bias { get; set; }
        //outgoing weights
        public double[] Weights { get; set; }
        public double Value { get; set; }
        public static Random random = new Random();

        public Neuron(int nextLayerSize)
        {
            this.Bias = random.NextDouble() *2.0 - 1.0;
            this.Value = 0.0;
            this.Weights = Enumerable.Range(0, nextLayerSize).Select(x => random.NextDouble() * 2.0 - 1.0).ToArray();
        }

        [JsonConstructor]
        public Neuron(double bias, double value, double[] weights)
        {
            Bias = bias;
            Value = value;
            Weights = weights;
        }

        public double Activate(double val)
        {
            return Math.Tanh(val);
        }

        public void Mutate(double chanceOfMutation, double AmplitudeOfMutation)
        {

            for (var wi = 0; wi < Weights.Length; wi++)
            {
                //chance should be between 0 and 1.
                var roll = random.NextDouble();
                if (roll < chanceOfMutation)
                {
                    Weights[wi] = (random.NextDouble() * 2.0 - 1.0) * AmplitudeOfMutation;
                }
            }

            //chance should be between 0 and 1.
            var roll2 = random.NextDouble();
            if (roll2< chanceOfMutation)
            {
                Bias = (random.NextDouble() * 2.0 - 1.0) * AmplitudeOfMutation;
            }

            return;
        }
    }

    public class Layer
    {
        public Neuron[] Neurons { get; set; }

        public Layer(int size, int nextLayerSize)
        {
            this.Neurons = Enumerable.Range(0, size).Select(x => new Neuron(nextLayerSize)).ToArray();
        }

        [JsonConstructor]
        public Layer(Neuron[] neurons)
        {
            this.Neurons = neurons;
        }

    }

    /// <summary>
    /// A simple Feed Forward Neural Network - 
    /// Assumes all layers are fully connected.
    /// </summary>
    public class NN
    {
        public Layer[] Layers { get; set; }

        public NN(int[] layerdimensions)
        {
            Layers = new Layer[layerdimensions.Length];
            var i = 0;
            foreach (var size in layerdimensions)
            {
                var nextLayerSize = layerdimensions.ElementAtOrDefault(i +1);
                Layers[i] = new Layer(size, nextLayerSize);
                i++;
            }
        }

        [JsonConstructor]
        public NN(Layer[] layers)
        {
            this.Layers = layers;
        }

        public double[] FF()
        {
            //foreach layer excluding the input layer
            for (int i = 1; i < Layers.Length; i++)
            {
                //foreach curneuron in the current layer
                var currentLayer = Layers[i];
                var prevLayer = Layers[i - 1];
                for (var j = 0; j < currentLayer.Neurons.Length; j++)
                {
                    var currentNeuron = currentLayer.Neurons[j];
                    //foreach prevneuron in the previous layer

                    double value = 0;
                    foreach (var prevNeuron in prevLayer.Neurons)
                    {
                        //accumulate a value for curNeuron using prevNeuron * prevNeuronWeight for all previous neurons.
                        value += (prevNeuron.Weights[j] * prevNeuron.Value);
                    }
                    //record final value for curNeuron using activation(value+CurNeuron.Bias)
                    currentNeuron.Value = currentNeuron.Activate(value + currentNeuron.Bias);
                }

            }
            return Layers[Layers.Length - 1].Neurons.Select(x => x.Value).ToArray();
        }

        public static string Serialize(NN instance)
        {
            var options = new System.Text.Json.JsonSerializerOptions(System.Text.Json.JsonSerializerDefaults.General);
            options.WriteIndented = true;
            return System.Text.Json.JsonSerializer.Serialize<NN>(instance, options);
        }

        public static NN Deserialize(string json)
        {
            return System.Text.Json.JsonSerializer.Deserialize<NN>(json);
        }

        public void Mutate(double chance, double amp)
        {
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Mutate(chance, amp);
                }
            }
        }

    }
}
