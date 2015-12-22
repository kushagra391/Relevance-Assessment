import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class FeaturesParser {

	private static String bm25Path = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//resultokapibm25.txt";
	private static String okapiPath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//resultokapitf.txt";
	private static String tfidfPath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//resulttfidf.txt";
	private static String jmPath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//resultunigramLMJM.txt";
	private static String laplacePath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//resultunigramLMWithLS.txt";
	private static String qrelPath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//qrels.adhoc.51-100.AP89.txt";
	private static String trainingARFFFilePath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//trainingARFFFile.arff";
	private static String testingARFFFilePath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//testingARFFFile.arff";
	private static String resultsFilePath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//predicted_results.txt";

	private static HashMap<Integer, HashMap<String, Double>> bm25 = new HashMap<Integer, HashMap<String, Double>>();
	private static HashMap<Integer, HashMap<String, Double>> okapi = new HashMap<Integer, HashMap<String, Double>>();
	private static HashMap<Integer, HashMap<String, Double>> tfidf = new HashMap<Integer, HashMap<String, Double>>();
	private static HashMap<Integer, HashMap<String, Double>> jm = new HashMap<Integer, HashMap<String, Double>>();
	private static HashMap<Integer, HashMap<String, Double>> laplace = new HashMap<Integer, HashMap<String, Double>>();

	private static HashMap<Integer, TestEntry> lookupTable = new HashMap<Integer, TestEntry>();

	private static int QUERYID_LOCATION = 0;
	private static int DOCID_LOCATION = 2;
	private static int SCORE_LOCATION = 4;

	private static int QREL_QUERYID_LOCATION = 0;
	private static int QREL_DOCID_LOCATION = 2;
	private static int QREL_ASSESSMENT_LOCATION = 3;

	public static void main(String[] args) throws Exception {

		// 1. prepare training and test sets
		ArrayList<Integer> trainingSet = getTrainingSet();
//		ArrayList<Integer> testSet = getTestSet(trainingSet);
		ArrayList<Integer> testSet = trainingSet;
//		System.out.println("TrainingSet created.");
//		System.out.println("TestSet created." );

		// 2. parse data/feature files
		// use clear() in case of memory issues. [very very bad approach
		// though.]
		parseDataFile(bm25Path, bm25, "BM25");
		parseDataFile(okapiPath, okapi, "Okapi");
		parseDataFile(tfidfPath, tfidf, "TF-IDF");
		parseDataFile(jmPath, jm, "JM");
		parseDataFile(laplacePath, laplace, "LSm");

		// 3. parse the QREL file
		// can be done in memory if other maps are not overloaded.
		createTrainingFile(trainingSet);
		createTestingFile(testSet);

		// 4. Start the Engine
		// Train
		DataSource source = new DataSource(trainingARFFFilePath);
		Instances trainDataSet = source.getDataSet();
		trainDataSet.setClassIndex(trainDataSet.numAttributes() - 1);
		// Creating a classifier
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(trainDataSet);

//		System.out.println(lr);

		// Test
		DataSource target = new DataSource(testingARFFFilePath);
		Instances testDataSet = target.getDataSet();
		testDataSet.setClassIndex(testDataSet.numAttributes() - 1);

		int size = testDataSet.numInstances();
//		System.out.println("testDataSet numInstances: " + size);
//		System.out.println("Loopup Table size: " + lookupTable.size());

		// int index = 0;
		HashMap<Integer, ArrayList<DocScore>> results = new HashMap<Integer, ArrayList<DocScore>>();
		for (int index = 0; index < size; index++) {

			TestEntry te = lookupTable.get(index);
			int queryID = te.queryID;
			String docID = te.docID;

			Instance instance = testDataSet.instance(index);
			double score = lr.classifyInstance(instance);

			// update the result
			if (results.containsKey(queryID)) {
				ArrayList<DocScore> entries = results.get(queryID);
				DocScore ds = new DocScore(docID, score);
				entries.add(ds);

				results.put(queryID, entries);
			} else {
				ArrayList<DocScore> entries = new ArrayList<DocScore>();
				DocScore ds = new DocScore(docID, score);
				entries.add(ds);

				results.put(queryID, entries);
			}
		}

		sortResults(results);
		storeSortedResults(results);
		System.out.println(lr);
		System.out.println("DONE!");
	}

	private static void storeSortedResults(
			HashMap<Integer, ArrayList<DocScore>> sortedResults)
			throws IOException {

		System.out.print("Storing results to file.");
		File file = new File(resultsFilePath);
		PrintWriter pr = new PrintWriter(new FileWriter(file));

		StringBuilder writeString = new StringBuilder();
		for (int queryID : sortedResults.keySet()) {

			int rank = 1;
			String qid = "" + queryID;
			ArrayList<DocScore> queryResults = sortedResults.get(queryID);

			for (DocScore ds : queryResults) {

				String docID = ds.docID;
				double _score = ds.score;
				String score = "" + _score;
				String rankString = "" + rank;

				String space = " ";
				String insertString = qid + space + "Q0" + space + docID
						+ space + rankString + space + score + space + "Exp";

//				System.out.println(insertString); // write to file
				writeString.append(insertString);
				writeString.append("\n");
				rank++;
			}

			pr.println(writeString);

		}
		System.out.println("\tDone");
	}

	private static void sortResults(
			HashMap<Integer, ArrayList<DocScore>> results) {
		
		System.out.print("Sorting Results.");
		for (int queryID : results.keySet()) {

			ArrayList<DocScore> queryResults = results.get(queryID);

			DocScoresComparator sortComp = new DocScoresComparator();
			queryResults.sort(sortComp);
		}

		System.out.println("\t\tDone");
	}

	private static void createTestingFile(ArrayList<Integer> testSet)
			throws IOException {

		System.out.print("Creating TestingFile");
		File arff = new File(testingARFFFilePath);
		PrintWriter pr = new PrintWriter(new FileWriter(arff));
		insertFileHeader(pr);

		try (BufferedReader br = new BufferedReader(new FileReader(qrelPath))) {
			String line;
			// System.out.println("qrelPath");
			int count = 0;
			while ((line = br.readLine()) != null) {
				// process the line.
				String[] tokens = line.split(" ");

				String _queryID = tokens[QREL_QUERYID_LOCATION];
				int queryID = Integer.parseInt(_queryID);
				if (!testSet.contains(queryID)) {// skip !trainingSet
					continue;
				}

				String _rating = tokens[QREL_ASSESSMENT_LOCATION];
				int rating = Integer.parseInt(_rating);
				String docID = tokens[QREL_DOCID_LOCATION];

				// extract features
				double f1, f2, f3, f4, f5;
				int label;

				try {
					f1 = bm25.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f1 = 0;
				}

				try {
					f2 = okapi.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f2 = 0;
				}

				try {
					f3 = tfidf.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f3 = 0;
				}

				try {
					f4 = jm.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f4 = 0;
				}

				try {
					f5 = laplace.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f5 = 0;
				}

				String _label = "?";

				String delim = ",";
				String insertString = f1 + delim + f2 + delim + f3 + delim + f4
						+ delim + f5 + delim + _label;
				pr.println(insertString);
				// System.out.println(insertString);

				// update the lookup table
				TestEntry te = new TestEntry(queryID, docID);
				lookupTable.put(count, te);
				count++;
			}
		}
		System.out.println("\t\tDone");
		pr.close();

	}

	private static void createTrainingFile(ArrayList<Integer> trainingSet)
			throws IOException {
		System.out.print("Creating TrainingFile");
		File arff = new File(trainingARFFFilePath);
		PrintWriter pr = new PrintWriter(new FileWriter(arff));
		insertFileHeader(pr);

		try (BufferedReader br = new BufferedReader(new FileReader(qrelPath))) {
			String line;
			// System.out.println("qrelPath");
			while ((line = br.readLine()) != null) {
				// process the line.
				String[] tokens = line.split(" ");

				String _queryID = tokens[QREL_QUERYID_LOCATION];
				int queryID = Integer.parseInt(_queryID);
				if (!trainingSet.contains(queryID)) {// skip !trainingSet
					continue;
				}

				String _rating = tokens[QREL_ASSESSMENT_LOCATION];
				int rating = Integer.parseInt(_rating);
				String docID = tokens[QREL_DOCID_LOCATION];

				// extract features
				double f1, f2, f3, f4, f5;
				int label;

				try {
					f1 = bm25.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f1 = 0;
				}

				try {
					f2 = okapi.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f2 = 0;
				}

				try {
					f3 = tfidf.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f3 = 0;
				}

				try {
					f4 = jm.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f4 = 0;
				}

				try {
					f5 = laplace.get(queryID).get(docID);
				} catch (NullPointerException e) {
					f5 = 0;
				}

				label = rating;

				String delim = ",";
				String insertString = f1 + delim + f2 + delim + f3 + delim + f4
						+ delim + f5 + delim + label;
				pr.println(insertString);
				// System.out.println(insertString);

			}
		}
		System.out.println("\t\tDone");
		pr.close();
	}

	private static void insertFileHeader(PrintWriter pr) {

		pr.println("@RELATION scores");
		pr.println("");

		// pr.println("@ATTRIBUTE totaltf        NUMERIC");
		// pr.println("@ATTRIBUTE totalidf        NUMERIC");
		// pr.println("@ATTRIBUTE doclength      NUMERIC");
		pr.println("@ATTRIBUTE okapibm25score NUMERIC");
		pr.println("@ATTRIBUTE okapitfscore   NUMERIC");
		pr.println("@ATTRIBUTE tfidfscore     NUMERIC");
		pr.println("@ATTRIBUTE lmjmscore       NUMERIC");
		pr.println("@ATTRIBUTE lmlaplacescore     NUMERIC");
		pr.println("@ATTRIBUTE score          REAL");
		pr.println("");

		pr.println("@Data");

	}

	private static void parseDataFile(String filePath,
			HashMap<Integer, HashMap<String, Double>> featureMap,
			String featureName) throws FileNotFoundException, IOException {

		System.out.print("Creating map for " + featureName);
		try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
			String line;
			// int count = 0;
			while ((line = br.readLine()) != null) {
				// process the line.
				String[] tokens = line.split(" ");
				String _queryID = tokens[QUERYID_LOCATION];
				String _score = tokens[SCORE_LOCATION];

				int queryID = Integer.parseInt(_queryID);
				Double score = Double.parseDouble(_score);
				String docID = tokens[DOCID_LOCATION];

				insertToMap(featureMap, queryID, docID, score);
				// System.out.println(count++);
			}
		}

		System.out.println("\t\tDone");

	}

	private static void insertToMap(
			HashMap<Integer, HashMap<String, Double>> map, int queryID,
			String docID, Double score) {

		if (map.containsKey(queryID)) {

			HashMap<String, Double> innerMap = map.get(queryID);
			innerMap.put(docID, score);

		} else {
			// new queryID found
			HashMap<String, Double> innerMap = new HashMap<String, Double>();
			map.put(queryID, innerMap);
		}

	}

	private static ArrayList<Integer> getTestSet(ArrayList<Integer> trainingSet) {

		ArrayList<Integer> list = new ArrayList<Integer>();
		ArrayList<Integer> originalList = getList();

		for (int i : originalList) {
			if (!trainingSet.contains(i)) {
				list.add(i);
			}
		}

		return list;
	}

	private static ArrayList<Integer> getTrainingSet() {

		// get a list of random number between 1-25
		ArrayList<Integer> list = getList();
		Collections.shuffle(list);

		// select a list of 20 numbers
		ArrayList<Integer> trainingSet = new ArrayList<Integer>();
		for (int i = 0; i < 20; i++) {
			trainingSet.add(list.get(i));
		}

		return trainingSet;
	}

	private static ArrayList<Integer> getList() {

		ArrayList<Integer> list = new ArrayList<Integer>();

		list.add(85);
		list.add(59);
		list.add(56);
		list.add(71);
		list.add(64);

		list.add(62);
		list.add(93);
		list.add(99);
		list.add(58);
		list.add(77);

		list.add(54);
		list.add(87);
		list.add(94);
		list.add(100);
		list.add(89);

		list.add(61);
		list.add(95);
		list.add(68);
		list.add(57);
		list.add(97);

		list.add(98);
		list.add(60);
		list.add(80);
		list.add(63);
		list.add(91);

		return list;
	}

}
