import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;

/**
 * Implementation of Delta TF-IDF word vectorization as described in the following paper.
 *
 * Paltoglou, G., and Thelwall, M "A study of Information Retrieval weighting schemes for sentiment analysis"
 */
enum TfType {
    NATURAL,
    LOGARITHM,
    AUGMENTED,
    BOOLEAN
}

enum IdfType {
    NORMAL_IDF,
    PROB,
    BM25,

    DELTA,   // Generates -inf and +inf
    DELTA_SMOOTHED,
    DELTA_PROB,
    DELTA_PROB_SMOOTHED
}

public class DeltaTfIdf {
    private static final String SEPARATOR = " ";
    private File class1;
    private File class2;

    public TreeMap<String, IdfData> corpusData;
    public List<Document> documents;
    private int class1Total;
    private int class2Total;

    public DeltaTfIdf(String fileClass1, String fileClass2) throws IOException {
        this(new File(fileClass1), new File(fileClass2));
    }

    public DeltaTfIdf(String fileClass1, String fileClass2, TfType tfType, IdfType idfType) throws IOException {
        this(new File(fileClass1), new File(fileClass2), tfType, idfType);
    }

    public DeltaTfIdf(File class1, File class2) throws IOException {
        this(class1, class2, TfType.NATURAL, IdfType.DELTA_SMOOTHED);
    }

    public DeltaTfIdf(File class1, File class2, TfType tfType, IdfType idfType) throws IOException {
        this.class1 = class1;
        this.class2 = class2;

        corpusData = new TreeMap<String, IdfData>();
        documents = new LinkedList<Document>();
        class1Total = 0;
        class2Total = 0;

        buildCorpusData();
        buildCorpusIdf(idfType);
        buildCorpusTf(tfType);
        buildCorpusTfIdf();
    }

    private void buildCorpusData() throws IOException {
        BufferedReader class1BufferedReader = new BufferedReader(new FileReader(class1));
        BufferedReader class2BufferedReader = new BufferedReader(new FileReader(class2));

        String tmpLine;

        while ((tmpLine = class1BufferedReader.readLine()) != null) {
            if (tmpLine.length() == 0)
                continue;
            Document document = new Document();
            this.documents.add(document);
            document.belongsToClass = 1;

            String[] terms = tmpLine.split(SEPARATOR);
            for (String term : terms) {
                IdfData corpusTermIdf;
                TfData documentTermTfData;
                if (!corpusData.containsKey(term)) {
                    corpusTermIdf = new IdfData();
                    corpusData.put(term, corpusTermIdf);
                } else
                    corpusTermIdf = corpusData.get(term);

                corpusTermIdf.corpus1Occurrence++;
                class1Total++;

                if (!document.documentData.containsKey(term)) {
                    documentTermTfData = new TfData();
                    document.documentData.put(term, documentTermTfData);
                } else
                    documentTermTfData = document.documentData.get(term);

                documentTermTfData.termOccurrence++;
                document.documentTotal++;
            }
        }

        while ((tmpLine = class2BufferedReader.readLine()) != null) {
            if (tmpLine.length() == 0)
                continue;
            Document document = new Document();
            this.documents.add(document);
            document.belongsToClass = 2;

            String[] terms = tmpLine.split(SEPARATOR);
            for (String term : terms) {
                IdfData corpusTermIdf;
                TfData documentTermTfData;
                if (!corpusData.containsKey(term)) {
                    corpusTermIdf = new IdfData();
                    corpusData.put(term, corpusTermIdf);
                } else
                    corpusTermIdf = corpusData.get(term);

                corpusTermIdf.corpus2Occurrence++;
                class2Total++;

                if (!document.documentData.containsKey(term)) {
                    documentTermTfData = new TfData();
                    document.documentData.put(term, documentTermTfData);
                } else
                    documentTermTfData = document.documentData.get(term);

                documentTermTfData.termOccurrence++;
                document.documentTotal++;
            }
        }
        class1BufferedReader.close();
        class2BufferedReader.close();
    }

    private void buildCorpusIdf(IdfType idfType) {
        for (String term : corpusData.keySet()) {
            IdfData termIdfData = corpusData.get(term);
            double numerator = 0.0;
            double denominator = 0.0;

            switch (idfType) {
                case DELTA:
                    numerator = (double) class1Total * (double) termIdfData.corpus2Occurrence;
                    denominator = (double) class2Total * (double) termIdfData.corpus1Occurrence;
                    break;
                case DELTA_SMOOTHED:
                    numerator = ((double) class1Total * (double) termIdfData.corpus2Occurrence) + 0.5;
                    denominator = ((double) class2Total * (double) termIdfData.corpus1Occurrence) + 0.5;
                    break;
                case NORMAL_IDF:
                    numerator = class1Total + class2Total;
                    denominator = termIdfData.corpus1Occurrence + termIdfData.corpus2Occurrence;
                    break;
                case PROB:
                    denominator = termIdfData.corpus1Occurrence + termIdfData.corpus2Occurrence;
                    numerator = (class1Total + class2Total) - denominator;
                    break;
                case BM25:
                    denominator = 0.5 + termIdfData.corpus1Occurrence + termIdfData.corpus2Occurrence;
                    numerator = 1.0 + (class1Total + class2Total) - denominator;
                    break;
                case DELTA_PROB:
                    numerator = (double)(class1Total - termIdfData.corpus1Occurrence) * (double)termIdfData.corpus2Occurrence;
                    denominator = (double)(class2Total - termIdfData.corpus2Occurrence) * (double)termIdfData.corpus1Occurrence;
                    break;
                case DELTA_PROB_SMOOTHED:
                    numerator = (double)(class1Total - termIdfData.corpus1Occurrence) * (double)termIdfData.corpus2Occurrence + 0.5;
                    denominator = (double)(class2Total - termIdfData.corpus2Occurrence) * (double)termIdfData.corpus1Occurrence + 0.5;
                    break;
            }

            termIdfData.totalIdf = Math.log10(numerator / denominator);
        }
    }

    private void buildCorpusTf(TfType tfType) {
        for (Document document : documents) {
            for (String term : document.documentData.keySet()) {
                TfData termTfData = document.documentData.get(term);

                double termTf = (double) termTfData.termOccurrence / (double) document.documentTotal;

                switch (tfType) {
                    case NATURAL:
                        termTfData.tf = termTf;
                        break;
                    case LOGARITHM:
                        termTfData.tf = 1 + Math.log10(termTf);
                        break;
                    case BOOLEAN:
                        termTfData.tf = termTf > 0 ? 1 : 0;
                        break;
                    case AUGMENTED:
                        termTfData.tf = termTf;
                        IdfData idfData = corpusData.get(term);
                        idfData.maxTf = Math.max(termTf, idfData.maxTf);
                        break;
                }
            }
        }

        if (tfType == TfType.AUGMENTED)
            for (Document document : documents) {
                for (String term : document.documentData.keySet()) {
                    TfData termTfData = document.documentData.get(term);
                    double maxTf = corpusData.get(term).maxTf;
                    double termTf = termTfData.tf;

                    termTfData.tf = 0.5 + ((0.5 * termTf) / maxTf);
                }
            }
    }

    private void buildCorpusTfIdf() {
        for (Document document : documents) {
            for (String term : document.documentData.keySet()) {
                TfData termTfData = document.documentData.get(term);
                double termIdf = corpusData.get(term).totalIdf;

                termTfData.tfIdf = termTfData.tf * termIdf;
            }
        }
    }

    class IdfData {
        private int corpus1Occurrence = 0;
        private int corpus2Occurrence = 0;
        private double maxTf = Double.MIN_VALUE;
        public double totalIdf = 0.0;
    }

    class Document {
        public TreeMap<String, TfData> documentData = new TreeMap<String, TfData>();
        private int documentTotal = 0;
        public int belongsToClass = 0;
    }

    class TfData {
        private int termOccurrence = 0;
        public double tf = 0.0;
        public double tfIdf = 0.0;
    }
}
