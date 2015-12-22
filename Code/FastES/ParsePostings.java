import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;

import javax.print.Doc;

public class ParsePostings {

	static int LIMIT = 84678;
	
	public static void main(String[] args) throws IOException {
		
		
		
		/*
		 * 1. Prepare query													-- DONE
		 * 2. Start the map engine											-- DONE
		 * 3. Process the rankings (take constants as constants for now)	-- DONE
		 * 4. Print
		 */
		
		createMap_ID();
		
		HashMap<String, String> map_for_ids = createMap_ID();
		System.out.println(map_for_ids.get("1"));
		
		
		
		HashMap<String, Long> map = createMap_DocLength();
		System.out.println(map.get("2"));
		
		System.out.println("Done !");
		
		
		
		
		ArrayList<String[]> runnable_queries = generateRunnableQueries();
		ArrayList<String> docIds = generateDocIds();
		
		
		//String[] query = runnable_queries.get(1);
		
		String query[] = { "report" };
 		
		System.out.println("START! ");
		//ArrayList<DocumentStats> results = OkapiTF.rankDocIdsForQuery_Okapi(query, docIds);
		//ArrayList<DocumentStats> results = TF_IDF.rankDocIdsForQuery_TF_IDF(query, docIds);
		//ArrayList<DocumentStats> results = Okapi_BM25.rankDocIdsForQuery_Okapi_BM25(query, docIds);
		ArrayList<DocumentStats> results = LaplaceSmoothing.rankDocIdsForQuery_Okapi(query, docIds);
		//ArrayList<DocumentStats> results = JMSmoothing.rankDocIdsForQuery_Okapi(query, docIds);
		ArrayList<DocumentStats> top100_results = rankTop100_Docs(results);
		
		// printTop100(top100_results);
		_postProcessResults("25", top100_results, map_for_ids);
		
		System.out.println("Result Size: " + results.size());
		
		
		
		
		
		/*
		String term = "performance"; 
		String docID = "1";
		
		HashMap<String, String> map_tf = getMap_TF();
		HashMap<String, String> map_df = getMap_DF();
		
		String tf = getTermPosting_tf(term, docID, map_tf);
		String df = getTermPosting_df(term, docID, map_df);
		System.out.println("tf: " + tf);
		System.out.println("df: " + df);
		*/
		
	}
	
	
	
	private static HashMap<String, String> createMap_ID() throws IOException {
		
		String path = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//Cache//docIDs";
		String content = new String(Files.readAllBytes(Paths.get(path)));
		
		String[] docIDs = content.split("\n");
		
		HashMap<String, String> map = new HashMap<String, String>();
		
		int count = 1;
		for (String docID : docIDs) {
			String key = "" + count;
			count++;
			map.put(key, docID);
		}
		
		return map;
		
	}
	
	
	
	private static void _postProcessResults(String q,
			ArrayList<DocumentStats> rankedScores, HashMap<String, String> map_for_ids) {
		// TODO Auto-generated method stub

		String queryNumber = q;
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
			docNo = map_for_ids.get("" + rankedScores.get(i).docID);

			entry = queryNumber + " " + q0 + " " + docNo + " " + rank + " "
					+ score + " " + exp;
			System.out.println(entry);
			entries.add(entry);
		}

		// System.out.println(entries.toString());
	}
	
	
	private static void printTop100(ArrayList<DocumentStats> top100_results) {
		// TODO Auto-generated method stub
		
		for (DocumentStats ds : top100_results) {
			System.out.println(ds);
		}
		
		
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
	
	
	
	
	private static ArrayList<String> generateDocIds() {
		// TODO Auto-generated method stub
		
		ArrayList<String> docIds = new ArrayList<String>();
		
		for (int i=1; i<100; i++) {
			String s = "" + i;
			docIds.add(s);
		}
		
		return docIds;
	}


	private static ArrayList<String[]> generateRunnableQueries() throws IOException {
		// TODO Auto-generated method stub
		
		ArrayList<String> qBucket = getQueryBucket();

		String[] queries = parseQueryFile();
		ArrayList<String[]> runnable_queries = new ArrayList<String[]>();

		for (String q : queries) {
			String[] query = cleanQuery(q, qBucket);
			runnable_queries.add(query);
		}

		//System.out.println(Arrays.toString(runnable_queries.get(1)));
		//System.out.println(queries.length);
		//System.out.println(runnable_queries.size());
		
		return runnable_queries;
	}


	private static String[] cleanQuery(String q, ArrayList<String> qBucket) {
		// TODO Auto-generated method stub
		
		String queryTerms[] = q.split(" ");
		
		ArrayList<String> terms = new ArrayList<String>();
		
		for (String s : queryTerms) {
			if (qBucket.contains(s)){
				terms.add(s);
			}
				
		}
		
		String returnQuery[] = new String[terms.size()];
 		
		int i=0;
		for (String term : terms) {
			returnQuery[i++] = term;
		}
		
		return returnQuery;
	}


	public static HashMap<String, String> getMap_TF() throws IOException {
		HashMap<String, String> map_tf = new HashMap<String,String>();
		ArrayList<String> bucket = getQueryBucket();		
		
		for (String b : bucket) {
			String tf = Term_tf(b);
			map_tf.put(b, tf);
		}
		
		return map_tf;
	}
	
	public static HashMap<String, String> getMap_DF() throws IOException {
		HashMap<String, String> map_df = new HashMap<String,String>();
		ArrayList<String> bucket = getQueryBucket();		
		
		for (String b : bucket) {
			String tf = Term_df(b);
			map_df.put(b, tf);
		}
		
		return map_df;
	}
	
	

	
	static String getTermPosting_df(String term, String docID,
			HashMap<String, String> map_df) {
		// TODO Auto-generated method stub
		
		return map_df.get(term).trim();
	}


	public static String getTermPosting_tf(String term, String docID,
			HashMap<String, String> map) {
		
		String tf_String = map.get(term);
		String tf[] = tf_String.split("\n");
	
		int pos = Integer.parseInt(docID);
		
		return tf[pos-1];
	}
	
	
	public static long getTermLength(String docID, HashMap<String, Long> map) {
		long doc_len;
		doc_len = map.get(docID);
		
		return doc_len;
	}


	private static ArrayList<String> getQueryBucket() throws IOException {
		// TODO Auto-generated method stub
		
		HashSet<String> bucket = new HashSet<String>();
		String Q[] = parseQueryFile();
		for (String q : Q) {
			// System.out.println(q);
			
			String terms[] = q.split(" ");
			for (String t : terms) {
				if (t.contains("dual"))
					continue;
				if (t.contains("down"))
					continue;
				bucket.add(t);
			}
		}

		removeStops(bucket);
		
		
		ArrayList<String> queryBucket = new ArrayList<String>();
		for (String t : bucket) {
			queryBucket.add(t);
		}
 		
		return queryBucket;
	}


	private static void removeStops(HashSet<String> bucket) throws IOException {
		// TODO Auto-generated method stub
		
		String[] stops = getStopWords();
		
		for (String stop : stops) {
			bucket.remove(stop);
		}
		
		bucket.remove("Document");
		bucket.remove("");
		bucket.remove(" ");
		
	}


	public static long Doc_Lenth(String docID) throws IOException {
		
		String content = readDocLengths();
		String contents[] = content.split("\n");
		
		int pos = Integer.parseInt(docID);
		String docLenth = contents[pos-1].trim();
		long length = Long.valueOf(docLenth).longValue();
		
		return length;
	}
	
	
	public static String readDocLengths() throws IOException {
		String path = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//Cached//docLengths.txt";
		String content = new String(Files.readAllBytes(Paths.get(path)));
		
		return content;
	}
	
	
	
	public static HashMap<String, Long> createMap_DocLength() throws IOException{
		String docLengths = readDocLengths();
		String contents[] = docLengths.split("\n");
	
		HashMap<String, Long> map = new HashMap<String, Long>();
		
		int count = 0; 
		for (String s : contents){
			count++;
			String docID = "" + count;
			long docLength = Long.valueOf(s.trim()).longValue();
			
			map.put(docID, docLength);
		}
		
		return map;
	}
	
	
	
	
	public static String Term_tf(String term) throws IOException {
		String defaultPath = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//QueryPostings//tf_";
		String path = defaultPath + term;
		
		String content = new String(Files.readAllBytes(Paths.get(path)));
		// System.out.println(content);
		
		return content;
		
	}
	
	public static String Term_df(String term) throws IOException {
		String defaultPath = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//QueryPostings//df_";
		String path = defaultPath + term;
		
		String content = new String(Files.readAllBytes(Paths.get(path)));
		// System.out.println(content);
		
		return content;
		
	}
	
	
	
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
	
	
	public static String[] getStopWords() 
			throws IOException {
		// TODO Auto-generated method stub
		String pathName = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//stoplist.txt";
		String content = new String(Files.readAllBytes(Paths.get(pathName)));
		
		String tokens[] = content.split("\n");
		
		return tokens;
	}
}
