import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.lucene.misc.GetTermInfo;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;


public class CashedPostings {
	
		
	
	
	public static void main(String[] args) throws ParseException, IOException {
		
		
		/*
		String term = "against";
		
		int SIZE = 50000; 
		String preURL = "http://localhost:9200/coffee5/_search?q=contents:";
		String urlPath = preURL + term + "&size=" + SIZE;
		String result = TermVectorStats.httpGet(urlPath);
		System.out.println(result.length());
				
		
		ArrayList<String> docIds = QueryRetriever.retrieveDocIDs(result);
		System.out.println(docIds.size());
		System.out.println(docIds.get(0));
		
		*/
		
		
		
		
		/*
		Long d = TermVectorStats.getDocLength("84075");
		System.out.println(d);
		*/
		
		
		HashSet<String> queryTerms = new HashSet<String>();
		
		String queries[] = QueryParser.parseQueryFile();
		// System.out.println(Arrays.toString(queries));
		
		
		for (String q : queries) {
			
			String Q[] = q.split(" ");
			
			for (String t : Q) {
				if (t.indexOf('"') != -1)
					continue;
				
				
				
				// t = t.replace("\"","");
				queryTerms.add(t);
				// System.out.println(t);
			}
		}
		
		// System.out.println(queryTerms.size());
		
		trimQueryTerms(queryTerms);
		System.out.println(queryTerms.size());
		
		for (String q : queryTerms) {
			// System.out.print(q + " \t--> ");
			
			storeQueryPostings(q);
			// System.out.println(FastStats.getTermPostings(q, "84672").tf);
		}
		
	
		
		
		
	
		
		
		
		
		
		
		/*
		String term = "Apple";
		ArrayList<String> docIds = new ArrayList<String>();
		for (int i=1; i<80000; i++) 
			docIds.add(""+i);
		
		
		
		long startTime = System.currentTimeMillis();
		System.out.println("Started");
		for (int i=1; i<79999; i++){
			TermPostings P = new TermPostings();
			P = FastStats.getTermPostings(term, docIds.get(i));
			//System.out.println(i);
		}
		System.out.println("Done");
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("elapsedTime: " + elapsedTime);
		
		*/
		
		
		
		/*
		
		String Q = "Document will discuss allegations, or measures being taken against, corrupt public officials of any governmental jurisdiction worldwide";
		String query[] = Q.split(" ");
		
		
		ArrayList<String> docIds = new ArrayList<String>();
		for (int i=1; i<80000; i++) 
			docIds.add(""+i);
		
		
		ArrayList<DocumentStats> ds = rankDocIdsForQuery_Okapi(query, docIds);
		System.out.println("Size: " + ds.size());
		
		*/
		
		/*
		ArrayList<String> docIds = new ArrayList<String>();
		for (int i=1; i<84679; i++) 
			docIds.add(""+i);
		
		
		for (String docID : docIds) {
			getAllPostings(docID);
		}
		*/
		
		
		/*
		ArrayList<String> docLengths = new ArrayList<String>();
		
		for (String docID : docIds) {
			String length = "" + TermVectorStats.getDocLength(docID);
			System.out.println("docLength for " + docID + "= " + length);
			docLengths.add("" + length);
		}
		
		
		PrintWriter writer = new PrintWriter("C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//the-file-name.txt", "UTF-8");
		
		for (String d : docLengths) {
			writer.println(d);
		}
		//writer.println("The first line");
		//writer.println("The second line");
		writer.close();
		*/
		
		
		
		
		
	}

	private static String trim(String term) {
		// TODO Auto-generated method stub
		
		term = term.replace("\"", "");
		term = term.replace(",", "");
		term = term.replace(".", "");
	
		return term;
	}

	private static void storeQueryPostings(String term) throws ParseException, IOException {
		// TODO Auto-generated method stub
		
		
		ArrayList<Long> tfs = new ArrayList<Long>();
		// Long df; 
		
		int SIZE = 50000; 
		String preURL = "http://localhost:9200/coffee5/_search?q=contents:";
		String urlPath = preURL + term + "&size=" + SIZE;
		String result = TermVectorStats.httpGet(urlPath);
				
		
		ArrayList<String> docIds = QueryRetriever.retrieveDocIDs(result);
		HashSet<String> docIDs = new HashSet<String>();
		docIDs.addAll(docIds);
		System.out.println(term + "--> \t" + docIds.size());
		
		
		for (int i=1; i < 84679; i++) {
			String docID = "" + i;
			
			if (docIDs.contains(docID)) {
				Long tf = FastStats.getTermPostings(term, docID).tf;
				tfs.add(tf);
			} 
			else tfs.add((long) 0);
		}
		
		String r = "";
		
		for (Long t : tfs) {
			r = r + t + "\n";
		}
		
		
		
		PrintWriter writer = new PrintWriter("C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//Cache//tf_" + term, "UTF-8");
		writer.println(r);
		writer.close();
		
		/*
		PrintWriter writer2 = new PrintWriter("C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//Cache//df_" + term, "UTF-8");
		if(docIds.size() == 0) {
			df = (long) 0;
		} else {
			df = FastStats.getTermPostings(term, docIds.get(0)).df;
		}
		writer2.println(df);
		writer2.close();
		*/
		System.out.println(term + " added.");
		
		
		
	}

	private static void trimQueryTerms(HashSet<String> queryTerms) throws IOException {
		// TODO Auto-generated method stub
		
		
		ArrayList<String> stops = getStopWords();
		
		stops.add("Document"); 
		stops.add("against,"); 
		stops.add("against"); 
		
		
		// remove stopwords 
		for (String s :stops) {
			if (queryTerms.contains(s)) {
				queryTerms.remove(s);
			}
		}
		
		// other bad elements
		if (queryTerms.contains(""))
			queryTerms.remove("");
		if (queryTerms.equals(" "))
			queryTerms.remove(" ");
		
		
			
		
		
	}
	
	
	public static ArrayList<String> getStopWords() 
			throws IOException {
		// TODO Auto-generated method stub
		String pathName = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//stoplist.txt";
		String content = new String(Files.readAllBytes(Paths.get(pathName)));
		
		String tokens[] = content.split("\n");
		
		
		ArrayList<String> stops = new ArrayList<String>();
		
		for (String t : tokens) {
			stops.add(t);
		}
		
		return stops;
	}
	

	public static void getAllPostings(String docID) throws IOException {
		// TermPostings P = new TermPostings();

		String preURL = "http://localhost:9200/coffee5/axe_dataset/";
		String postURL = "/_termvector?pretty=true&term_statistics=true";
		String urlPath = preURL + docID + postURL; // check for bad ID 

		String result = TermVectorStats.httpGet(urlPath);
		
		PrintWriter writer = new PrintWriter("C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//Cache//" + docID, "UTF-8");
		writer.println(result);
		
		writer.close();
		
		
	}

	
	public static ArrayList<DocumentStats> rankDocIdsForQuery_Okapi(
			String[] query, ArrayList<String> docIds) throws ParseException, IOException {
		
		ArrayList<DocumentStats> rankedDocsForQuery = new ArrayList<DocumentStats>();
		
		// initialize rankedDocsForQuery
		for (String d : docIds) {
			DocumentStats ds = new DocumentStats(d, 0.0);
			rankedDocsForQuery.add(ds);
		}
		
		
		for (String term : query) {
			ArrayList<DocumentStats> newRankingOfDocs = rankDocIdsForTerms_Okapi(term, docIds);
			//ArrayList<DocumentStats> newRankingOfDocs = rankedDocsForQuery;
			addScores(newRankingOfDocs, rankedDocsForQuery);
		}
		
		return rankedDocsForQuery;
	}


	public static void addScores(ArrayList<DocumentStats> newRankingOfDocs,
			ArrayList<DocumentStats> rankedDocsForQuery) {
		// TODO Auto-generated method stub
		
		int s1 = newRankingOfDocs.size();
		int s2 = rankedDocsForQuery.size();
		
		int size; 
		if (s1 != s2) 
			System.out.println("FATAL ERROR");
		
		size = s1 = s2;
		
		
		for (int i=0; i<size; i++) {

			if (!rankedDocsForQuery.get(i).docID
					.equals(newRankingOfDocs.get(i).docID))
				System.out.println("FATAL ERROR");
				
			
			rankedDocsForQuery.get(i).score = rankedDocsForQuery.get(i).score
					+ newRankingOfDocs.get(i).score;
		}
		
	}


	public static ArrayList<DocumentStats> rankDocIdsForTerms_Okapi(
			String term, ArrayList<String> docIds) throws ParseException, IOException {
		// TODO Auto-generated method stub
		
		ArrayList<DocumentStats> rankedDocForTerm = new ArrayList<DocumentStats>();
		
		for (String doc : docIds) {
			DocumentStats ds = calculateDocScoreForTerm(term, doc);
			rankedDocForTerm.add(ds);
		}
		
		return rankedDocForTerm;
	}


	public static DocumentStats calculateDocScoreForTerm(String term,
			String docID) throws ParseException, IOException {
		// TODO Auto-generated method stub
		
		double score;
		
		long tf_wd; 
		long len_d;
		double avg_len_d;
		
		tf_wd = FastStats.getTermPostings(term, docID).tf;
		len_d = FastStats.getDocLength(docID);
		avg_len_d = len_d; // for now, cache the result for later use. 
		
		score = tf_wd / (tf_wd + 0.5 + 1.5 * (len_d / avg_len_d));
		
		DocumentStats ds = new DocumentStats(docID, score);
		return ds;
	}

	
}

class FastStats {
	
	
	static String dirPath = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//";
	
	public static TermPostings getTermPostings(String term, String docID) throws ParseException, IOException {
		
		TermPostings P = new TermPostings();
		
		String dir = "Cached//";
		String fileName = docID;
		
		String pathName = dirPath + dir + fileName;
		String result = new String(Files.readAllBytes(Paths.get(pathName)));
		
		
		// read the file with docID
		
		
		// json parse logic similar
		JSONParser parser = new JSONParser();
		Object obj = parser.parse(result);

		JSONObject obj2 = (JSONObject) obj;
		JSONObject obj3 = (JSONObject) obj2.get("term_vectors");
		JSONObject contents = (JSONObject) obj3.get("contents");
		if (contents == null) {
			P.df = 0;
			P.tf = 0;
			P.ttf = 0;
			return P;
		}
		JSONObject terms = (JSONObject) contents.get("terms");
		JSONObject token1 = (JSONObject) terms.get(term);
		
		// System.out.println(token1.keySet());
		// long count = terms.keySet().size();
		
		// System.out.println("Document length: " + count);
		
		
		// return 0, if the term is not present in the document 
		if (token1 == null) {
			P.df = 0;
			P.tf = 0;
			P.ttf = 0;
			return P;
		}
		
		// extract needed details 
		long tf19 = (long) token1.get("term_freq");
		long df19 = (long) token1.get("doc_freq");
		long ttf19 = (long) token1.get("ttf");
		
		// set parameters 
		P.tf = tf19;
		P.df = df19;
		P.ttf = ttf19;
		
		return P;
		
	}
	
	
	public static long getDocLength(String docID) throws ParseException, IOException {
		
		TermPostings P = new TermPostings();
		
		String dir = "Cached//";
		String fileName = docID;
		
		String pathName = dirPath + dir + fileName;
		String result = new String(Files.readAllBytes(Paths.get(pathName)));
		
		
		// read the file with docID
		
		
		// json parse logic similar
		JSONParser parser = new JSONParser();
		Object obj = parser.parse(result);

		JSONObject obj2 = (JSONObject) obj;
		JSONObject obj3 = (JSONObject) obj2.get("term_vectors");
		JSONObject contents = (JSONObject) obj3.get("contents");
		if (contents == null) {
			P.df = 0;
			P.tf = 0;
			P.ttf = 0;
			return 0;
		}
		JSONObject terms = (JSONObject) contents.get("terms");
		// JSONObject token1 = (JSONObject) terms.get(terms);
		
		// System.out.println(token1.keySet());
		long count = terms.keySet().size();
		
		return count;
		
	}
	
	
	
}


/*
class Box {
	long tf; // term frequency, term_freq (number of times the term occurs in the document)
	long df; // doc frequency, doc_freq # (the number of documents containing the current term)
	long ttf; // total term frequency, ttf # (how often a term occurs in all documents)
	
	// document details
	// sum_doc_freq, doc_count, sum_ttf # check if any such details are required. 
	
	
	public String toString() {
		return "tf: " + tf + ", " + "df: " + df + ", " + "ttf: " + ttf;
		
	}
	
}
*/