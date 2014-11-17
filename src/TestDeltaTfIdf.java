import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Generates an arff file compatible with Weka
 */
public class TestDeltaTfIdf {

    public static void main(String[] args){
        try {
            DeltaTfIdf deltaTfIdf = new DeltaTfIdf("pos", "neg", TfType.AUGMENTED, IdfType.DELTA_SMOOTHED); // Default values for TfType and IdfType


            PrintStream printStream = new PrintStream("output_5.arff");

            printStream.println("@RELATION tweets\n");

            HashMap<String, Integer> words = new HashMap<String, Integer>();
            int wordsCount = 0;

            for (String word : deltaTfIdf.corpusData.keySet()) {
                words.put(word, wordsCount++);
                printStream.println("@ATTRIBUTE a" + wordsCount + " REAL");
            }

            printStream.println("@ATTRIBUTE class {pos,neg}\n\n");

            printStream.println("\n@DATA");

            for(DeltaTfIdf.Document document : deltaTfIdf.documents){
                double[] vector = new double[wordsCount];

                Map<String, DeltaTfIdf.TfData> tf_idfMap = document.documentData;
                for(String word : tf_idfMap.keySet()){
                    int index = words.get(word);
                    double tf_idf = tf_idfMap.get(word).tfIdf;

                    vector[index] = tf_idf;
                }

                for(Double d : vector)
                    printStream.print(d + ",");

                String attribute = document.belongsToClass == 1 ? "pos" : "neg";
                printStream.println(attribute);
            }

            printStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
