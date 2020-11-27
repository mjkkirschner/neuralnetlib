using System;
using neuralnet1;

namespace NNCLI
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var nn = new neuralnet1.NN(new int[] { 3, 3, 3 });
            var json = NN.Serialize(nn);
            Console.WriteLine(json);
            var nn2 = NN.Deserialize(json);

            Console.WriteLine(NN.Serialize(nn2));


        }
    }
}
