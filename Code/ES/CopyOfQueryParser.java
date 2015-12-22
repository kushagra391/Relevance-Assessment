// import groovy.json.internal.ArrayUtils;

import java.awt.List;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.concurrent.PriorityBlockingQueue;

import org.elasticsearch.common.util.ArrayUtils;
import org.json.simple.parser.ParseException;


public class CopyOfQueryParser {
	
	String path = "";
	
	public static void main(String[] args) throws IOException, ParseException {
	
		
		/*
		LinkedList<DocumentStats> documentScores = addSomeScores();
		LinkedList<DocumentStats> rankedScores = rankTop100(documentScores);
		postProcessResults(null, rankedScores);
		*/
		
		String queries[] = parseQueryFile();
		System.out.println(Arrays.toString(queries));
		
		
		for (String q : queries) {
			System.out.println(q);
		}
		
		
		
		// list of queries found
		
		// for each query
		//		1. generate a list of documents to be queried 		? 
		//		2. select a retrieval model 						:)
		// 		3. call the functions to rank the documents		:)
		//		4. post process them into a file					:)
		
		
			
			ArrayList<String> docs = new ArrayList<String>();
			docs.addAll(createSomeDocList());
			
			ArrayList<DocumentStats> rankedDocs = OkapiTF.rankDocIdsForQuery_Okapi(queries, docs);
			ArrayList<DocumentStats> top100 = rankTop100_Docs(rankedDocs);
		
			//_postProcessResults(q, rankedScores);
		
		
		
		
		
		
		long startTime = System.currentTimeMillis();
		
		int count = 1;
		for (String q : queries) {
			
		//		1. generate a list of documents to be queried 		? 
			// LinkedList<String> docList = retrieveDocsList(q); 	
			
			ArrayList<String> docIDs = QueryRetriever.getDocListsFromQuery(q.split(" "));
			
			LinkedList<String> docList = createSomeDocList(); 		
			//LinkedList<String> docList = new LinkedList<String>();
			//docList.addAll(docIDs);
			String[] query = q.split(" ");
 			
		//		2. select a retrieval model 						:)
			LinkedList<Double> docScores = TermVectorStats.Okapi_Score(docList, query);			// 2
			
		// 		3. call the functions to rank the documents		:)
			LinkedList<DocumentStats> documentScores = addSomeScores();
			LinkedList<DocumentStats> rankedScores = rankTop100(documentScores);
			
		//		4. post process them into a file					:)
			postProcessResults(q, rankedScores);
			
			
			System.out.println("=================== Cycle " + count++ + " / " + queries.length + " ================");
			
			
		}

		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;

		System.out.println("elapsedTime: " + elapsedTime);
		System.out.println("elapsedTime per query: " + elapsedTime/queries.length);
	}

	
	private static LinkedList<String> createSomeDocList() {
		// TODO Auto-generated method stub
		
		LinkedList<String> docList = new LinkedList<String>();
		
		for (int i=1; i<100 ; i++) {
			docList.add("" + i);
		}
		
		return docList;
	}


	// get queryNumber
	private static void postProcessResults(String q,
			LinkedList<DocumentStats> rankedScores) {
		// TODO Auto-generated method stub

		String queryNumber = "" + -1;
		String q0 = "Q0";
		String docNo;
		String rank;
		String score;
		String exp = "Exp";

		String entry;
		ArrayList<String> entries = new ArrayList<String>(rankedScores.size());
		for (int i = 0; i < rankedScores.size(); i++) {
			// System.out.println(rankedScores.get(i));
			rank = "" + i;
			score = "" + rankedScores.get(i).score;
			docNo = "" + rankedScores.get(i).docID;

			entry = queryNumber + " " + q0 + " " + docNo + " " + rank + " "
					+ score + " " + exp;
			System.out.println(entry);
			entries.add(entry);
		}

		// System.out.println(entries.toString());
	}

	private static void _postProcessResults(String q,
			LinkedList<DocumentStats> rankedScores) {
		// TODO Auto-generated method stub

		String queryNumber = "" + -1;
		String q0 = "Q0";
		String docNo;
		String rank;
		String score;
		String exp = "Exp";

		String entry;
		ArrayList<String> entries = new ArrayList<String>(rankedScores.size());
		for (int i = 0; i < rankedScores.size(); i++) {
			// System.out.println(rankedScores.get(i));
			rank = "" + i;
			score = "" + rankedScores.get(i).score;
			docNo = "" + rankedScores.get(i).docID;

			entry = queryNumber + " " + q0 + " " + docNo + " " + rank + " "
					+ score + " " + exp;
			System.out.println(entry);
			entries.add(entry);
		}

		// System.out.println(entries.toString());
	}

	
	
	
	// temporary method for testing
	private static LinkedList<DocumentStats> addSomeScores() {
		// TODO Auto-generated method stub
		
		DocumentStats d1 = new DocumentStats("A1", 10.0);
		DocumentStats d2 = new DocumentStats("A2", 12.0);
		DocumentStats d3 = new DocumentStats("A3", 1.3);
		DocumentStats d4 = new DocumentStats("A4", 14.2);
		DocumentStats d5 = new DocumentStats("A5", 9.1);
		DocumentStats d6 = new DocumentStats("A7", 11.43);
		
		LinkedList<DocumentStats> D = new LinkedList<DocumentStats>();
		
		D.add(d1);
		D.add(d2);
		D.add(d3);
		D.add(d4);
		D.add(d5);
		D.add(d6);
		
		return D;
	}


	private static LinkedList<DocumentStats> rankTop100(LinkedList<DocumentStats> documentScores) {
		
		Comparator<DocumentStats> comparator = new DocumentStatsComparator(); 			// comparator
		PriorityQueue<DocumentStats> top100 = new PriorityQueue<DocumentStats>(comparator);
	
				
		// put all the docs in a min heap, define the comparator. #
		for (DocumentStats ds : documentScores) {
			top100.add(ds);
		}
		
		// retrieve the top 100 into result
		LinkedList<DocumentStats> result = new LinkedList<DocumentStats>(); 
		for (int i=0; i<100; i++) {
			DocumentStats r = top100.poll();
			
			if (r == null) {
				// if r is null --> documentScores.size < 100
				break;
			}
			
			result.add(r);
		}
		
		// return
		return result;
	}

	private static ArrayList<DocumentStats> rankTop100_Docs(
			ArrayList<DocumentStats> documentScores) {

		Comparator<DocumentStats> comparator = new DocumentStatsComparator(); // comparator
		PriorityQueue<DocumentStats> top100 = new PriorityQueue<DocumentStats>(
				comparator);

		// put all the docs in a min heap, define the comparator. #
		for (DocumentStats ds : documentScores) {
			top100.add(ds);
		}

		// retrieve the top 100 into result
		ArrayList<DocumentStats> result = new ArrayList<DocumentStats>();
		for (int i = 0; i < 100; i++) {
			DocumentStats r = top100.poll();

			if (r == null) {
				// if r is null --> documentScores.size < 100
				break;
			}

			result.add(r);
		}

		// return
		return result;
	}

	private static LinkedList<String> retrieveDocsList(String q) {
		// TODO Auto-generated method stub
		return null;
	}


	// RETURNS: 
	public static String[] parseQueryFile() throws IOException {
		
		String pathName = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//query_desc.51-100.short.txt";
		String content = new String(Files.readAllBytes(Paths.get(pathName)));
		
		String queries[] = content.split("\n");
 				
		// System.out.println(content);
		// System.out.println(Arrays.toString(queries));
		
		String Q[] = new String[queries.length];
		int index = 0;
		for (String d : queries) {
			int beginIndex = d.indexOf("Document");
			Q[index++] = d.substring(beginIndex);
		}
		
		// System.out.println(Arrays.toString(Q));
		
		return Q;
	}
	
	// RETURNS: trims unnecessary words from the queries
	public static void optimizeQueries() {
		
	}
	
	// Helper for optimizeQueries()
	public static String[] queryStopWords(String Q[]) {
		String[] stops = { "Document", "will" };
		
		return stops;
		
	}
	
	private static boolean termsContainsStop(String t, String[] stops) {
		// TODO Auto-generated method stub
		return false;
	}


	public static void QueryExpansion() {
		
	}
	
	
	
	
}
