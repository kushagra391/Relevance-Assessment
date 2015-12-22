import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

import javax.print.Doc;

import org.apache.lucene.codecs.TermStats;
import org.apache.lucene.index.DocsAndPositionsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.elasticsearch.action.termvector.TermVectorRequestBuilder;
import org.elasticsearch.action.termvector.TermVectorResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.node.Node;

/* Json */
import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.JSONValue;
import org.json.simple.parser.ParseException;
import org.json.simple.parser.JSONParser;

import com.vividsolutions.jts.util.Stopwatch;


public class TermVectorStats {
	final static String indexName = "coffee5";
	final static String documentType = "axe_dataset";
	final static String fieldName = "contents";
	
	public static void main(String[] args) throws IOException, ParseException {

		long startTime = System.currentTimeMillis();
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		// System.out.println("elapsedTime: " + elapsedTime);
		
		
		int a = 10;
		assert a == 10;
		
		
		
		/* calculate okapiTF */
		/*
		String term = "head";
		String query[] = {"the", "head", "always", "level" };		
		
		LinkedList<String> docList = new LinkedList<String>();
		for (int i=1; i<1000; i++) {
			docList.add(Integer.toString(i));
		}
		*/
		
		/* FRESH START */
		
		
		/*
		 * 1. Get a query -- String[]
		 * 2. Get the docIDs asscociated to the query -- ArrayList<String>, String = docID
		 * 3. Rank all the docIDs for one term 			-- ArrayList<DocumentStats>
		 * 4. Rank all the docIDs for <all the terms = query > -- ArrayList<DocumentStats>
		 */
		
		
		/* ======================= 1 ================================*/
		String allQueries[] = QueryParser.parseQueryFile();
		String Query = allQueries[1];
		String query[] = Query.split(" ");
		System.out.println("Query Tokens: " + Arrays.toString(query));
		
		/* ======================= 2 =============================== */
		ArrayList<String> docIds = QueryRetriever.getDocListsFromQuery(query);
		System.out.println("List of Relevany DocIds (size) " + docIds.size());
		
		
		/* TEMP */
		ArrayList<String> docIDs = new ArrayList<String>();
		for (int i=1; i<100; i++) 
			docIDs.add("" + i);
		
		
		
		/* ======================= 3 =============================== */
		String term = query[2];
		System.out.println("Term: " + term);
		ArrayList<DocumentStats> rankedDocsForTerm = OkapiTF.rankDocIdsForTerms_Okapi(term, docIDs);
		System.out.println(rankedDocsForTerm.size());
		
		for (DocumentStats ds : rankedDocsForTerm) {
			System.out.println("Term Stats: " + "docID: " + ds.docID + ", Score: " +  ds.score);
		}
		
		/* ======================= 4 =============================== */
		ArrayList<DocumentStats> rankedDocsForQuery = OkapiTF.rankDocIdsForQuery_Okapi(query, docIDs);
		System.out.println(rankedDocsForQuery.size());
	
		for (DocumentStats ds : rankedDocsForQuery) {
			System.out.println("Query Stats: " + "docID: " + ds.docID + ", Score: " +  ds.score);
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		/* FRESH START */
		
		
		
		
		
		
		
		
		
		
		
		
		/*
		// 1. OKAPI-TF
		System.out.println("Term Okapi score \t:" + Okapi_DocListScores(docList, term).toString());
		System.out.println("Query Okapi score \t:" + Okapi_Score(docList, query).toString());
		
		
		// 2. TF-IDF
		System.out.println("Term TF-IDF score \t:" + TF_IDF_DocListScores(docList, term).toString());
		System.out.println("Query TF-IDF score \t:" + TF_IDF_Score(docList, query).toString());
		
		// 3. OKAPI BM25
		System.out.println("Term OKAPI BM25 score\t:" + Okapi25_DocListScores(docList, term, query).toString());
		System.out.println("Query OKAPI BM25 score\t:" + Okapi25_Score(docList, query).toString());
	
		// 4. Unigram LM with Laplace smoothing::
		System.out.println("Term UnigramLM score\t:" + UnigramLM_DocListScores(docList, term, query).toString());
		System.out.println("Query UnigramLM score\t:" + UnigramLM_Score(docList, query).toString());
		
		// 5. Unigram LM with JelinekMercer Smoothing
		*/
		
		
		
		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	/* ======================= convert scores to Doc + Scores =============================== */
	
	public static LinkedList<DocumentStats> score2Stats(LinkedList<String> docList, LinkedList<Double> scores) {
		
		LinkedList<DocumentStats> ds = new LinkedList<DocumentStats>();
		
		int s1 = docList.size();
		int s2 = scores.size();
		if (s1 != s2) 
			System.out.println("##ERROR: Check lists");
		
		for (int i=0; i<s1; i++) {
			DocumentStats d = new DocumentStats(docList.get(i), scores.get(i));
			ds.add(d);
		}
		
		
		return ds; 
	}
	
	
	
	
	/*   =================================== Unigram LM =====================================  */
	
	public static LinkedList<Double> UnigramLM_Score(LinkedList<String> docList, String[] query) 
			throws ParseException, IOException {
		
		LinkedList<Double> docScores = new LinkedList<Double>();
		LinkedList<Double> termScores = new LinkedList<Double>();
		
		// initialize docScores
		for (int i=0; i<docList.size(); i++) {
			docScores.add((double) 0);
		}
		
		for (int i=0; i<query.length; i++) {
			termScores = UnigramLM_DocListScores(docList, query[i], query);
			docScores = updateList(termScores, docScores);
		}
		
		return docScores;
	}
	
	
	/*
	 * # Get the Vocabulary of the corpus
	 * Log giving negative values ?!
	 */
	public static LinkedList<Double> UnigramLM_DocListScores(
			LinkedList<String> docList, String term, String[] query) throws ParseException, IOException {
		// TODO Auto-generated method stub
		LinkedList<Double> docScores = new LinkedList<Double>();
		
		long tfwf;
		long doc_length;
		long V;
		
		V = 100;  // # for now
		
		
		double score;
		for (String d : docList) {
			doc_length = getDocLength(d);
			tfwf = getTermPostings(term, d).tf;
			
			// System.out.println(tfwf);
			//System.out.println((tfwf +1)/ (doc_length + V));
			score =  (double)(tfwf + 1) / (double)(doc_length + V) ;
			score = Math.log(score);
			
			docScores.add(score);
		}
		
		return docScores;
	}

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*   =================================== OKAPI 25 =====================================  */
	
	public static LinkedList<Double> Okapi25_Score(LinkedList<String> docList, String[] query) 
			throws ParseException, IOException {
		
		LinkedList<Double> docScores = new LinkedList<Double>();
		LinkedList<Double> termScores = new LinkedList<Double>();
		
		// initialize docScores
		for (int i=0; i<docList.size(); i++) {
			docScores.add((double) 0);
		}
		
		for (int i=0; i<query.length; i++) {
			termScores = Okapi25_DocListScores(docList, query[i], query);
			docScores = updateList(termScores, docScores);
		}
		
		return docScores;
	}

	
	/* calculate doc Scores for T for all documents int docList */
	/* 
	 * Issues:
	 * --> docLength is calculated fine. 
	 * --> But avgDocLength calculation is bad for now 
	 * --> Also, term frq in query. Does it needs to be analyzed further ?
	 * # This is a language model. So, ?  
	 * --> D - total number of documents in the corpus 
	 */
	public static LinkedList<Double> Okapi25_DocListScores(LinkedList<String> docList,
			String term, String query[]) throws ParseException, IOException {
		
		LinkedList<Double> docTermScores = new LinkedList<Double>();
		
		
		/* 
		 * tf(w,q)		-- freq of T in Q
		 * dfw 			-- doc_freq
		 * len(d)		-- length of the document
		 * avg(len(D))	-- avg length of the document in the corpus
		 */
		
		
		/* 
		 * word freuency in the query
		 * # stemming, language modesl etc ??  
		 */
		@SuppressWarnings("unused")
		int tf_wq = 0;
		for (String q : query) {
			if (term.equals(q)) {
				tf_wq++;
			}
		}
		
		
		/* doc length */ 
		@SuppressWarnings("unused")
		long doc_length; 
	
		
		/* avg doc length */
		// TODO : add this into a function
		long avg_doc_length = 0;
		for (String d : docList) {
			avg_doc_length = avg_doc_length + getDocLength(d);
		}
		avg_doc_length = avg_doc_length / docList.size() ; // calculated 
		
		long D; 
		D = docList.size() ; // for now, write a function later
		
		long dfw = 0;
		int count = 0; 
		while (dfw !=0 || count < docList.size()) {
			dfw = getTermPostings(term, docList.get(count)).df;
			count ++ ;
		} 
		
		if (dfw == 0) {
			// System.out.println("dfw is zero for term  " + term);
		}
		
		
		
		double term_score;
		for (String d : docList) {
			doc_length = getDocLength(d);
			long tfwd = getTermPostings(term, d).tf;
			term_score = calculateOkapiBM25_termScore(term, d, D, dfw, tfwd,
					tf_wq, doc_length, avg_doc_length);
			docTermScores.add(term_score);
		}
		
		
		return docTermScores;
	}

	public static double calculateOkapiBM25_termScore(String term, String d,
			long D, long dfw, long tfwd, int tf_wq, long doc_length,
			long avg_doc_length) {
		double score;
		
		double first;
		double second;
		double third;
		
		// take the values from the slide
		double k1 = 1;
		double k2 = 1;
		double b = 0.2;
		
		first = Math.log((D + 0.5) / (dfw + 0.5));
		// System.out.println("D, dfw : " + D + ", " + dfw);
		second =  (tfwd + k1 * tfwd) / (tfwd + k1 * ((1 - b) + b * (doc_length / avg_doc_length)));
		third = (tf_wq + k2 * tf_wq) / (tf_wq + k2) ;
		
		// System.out.println("Scores : " + first + ", " + second + ", " + third);
		
		score = first * second * third; 
		
		return score;
	}
	/*  End of OKAPI 25 */

	public static LinkedList<Double> TF_IDF_Score(LinkedList<String> docList, String[] query) 
			throws ParseException, IOException {
		
		LinkedList<Double> docScores = new LinkedList<Double>();
		LinkedList<Double> termScores = new LinkedList<Double>();
		
		// initialize docScores
		for (int i=0; i<docList.size(); i++) {
			docScores.add((double) 0);
		}
		
		for (int i=0; i<query.length; i++) {
			termScores = TF_IDF_DocListScores(docList, query[i]);
			docScores = updateList(termScores, docScores);
		}
		
		return docScores;
	}
	
	
	
	public static LinkedList<Double> Okapi_Score(LinkedList<String> docList, String[] query) 
			throws ParseException, IOException {
		
		LinkedList<Double> docScores = new LinkedList<Double>();
		LinkedList<Double> termScores = new LinkedList<Double>();
		
		// initialize docScores
		for (int i=0; i<docList.size(); i++) {
			docScores.add((double) 0);
		}
		
		for (int i=0; i<query.length; i++) {
			termScores = Okapi_DocListScores(docList, query[i]);
			docScores = updateList(termScores, docScores);
		}
		
		return docScores;
	}
	
	
	public static LinkedList<Double> updateList(LinkedList<Double> list1,
			LinkedList<Double> list2) {
		
		LinkedList<Double> result = new LinkedList<Double>();
		
		for (int i=0; i<list1.size(); i++) {
			result.add(list1.get(i) + list2.get(i));
		}
		
		return result;
	}


	/*
	 * Issues : 
	 * dfw is a constant. Keep note of that. Wrong implementation of that for now. 
	 * 
	 */
	public static LinkedList<Double> TF_IDF_DocListScores(LinkedList<String> docList,
			String term) throws ParseException, IOException {
		// TODO Auto-generated method stub

		LinkedList<Double> docScores = new LinkedList<Double>();

		long D = 80000;

		long df_w;
		double okapi_tf;

		double score = 0;
		for (int i = 0; i < docList.size(); i++) {
			df_w = getTermPostings(term, docList.get(i)).df;
			/* zero condition */
			if (df_w == 0) {
				docScores.add((double) 0);
				continue;
			}

			okapi_tf = calculate_doc_OkapiTF(term, docList.get(i));
			score = okapi_tf * (Math.log((double) D / df_w));
			docScores.add(score);
		}

		return docScores;
	}



	/* GIVEN: calculate the doc score, given a list of docs and a document */
	/* RETURNS: return the list of scores for all the documents */
	public static LinkedList<Double> Okapi_DocListScores(LinkedList<String> docList,
			String term) throws ParseException, IOException {
		
		LinkedList<Double> docScores = new LinkedList<Double>();
		
		
		// avgDocLength
		long docCount = docList.size();
		long totalDocLength = 0;
		
		for (int i=0; i<docList.size(); i++) {
			totalDocLength =+ getDocLength(docList.get(i));
		}
		
		double avgDocLength = totalDocLength / docCount;
		// avgDocLength calculated
		
		
		// start calculating scores
		double score;
		long tf;
		long docLenth;
		for (int i=0; i<docList.size(); i++) {
			tf = getTermPostings(term, docList.get(i)).tf;
			docLenth = getDocLength(docList.get(i));
			score = tf / (tf + 0.5 + 1.5 * (docLenth / avgDocLength));
			docScores.add(score);
		}
			
		return docScores;
		// return score2Stats(docList, docScores);
	}

	
	
	/* calculate the doc score, given a term and a document */
	/* RETURNS: a score for the document */
	public static double calculate_doc_OkapiTF(String term, String docID)
			throws ParseException, IOException {
		
		double score;

		/* 3 things:
		 * tfw,d
		 * len(d)
		 * avg(len(d))
		 */
		
		TermPostings P = getTermPostings(term, docID);
		long tf = P.tf;
		
		long docLength = getDocLength(docID);
		long avgDocLength = docLength; // for now
		
		score = (double) (tf / (tf + 0.5 + 1.5 * (docLength / avgDocLength)));
		
		return score;
	}

	// computes the document length
	public static long getDocLength(String docID) 
			throws ParseException, IOException {
		
		String preURL = "http://localhost:9200/coffee5/axe_dataset/";
		String postURL = "/_termvector?pretty=true&term_statistics=true";
		String urlPath = preURL + docID + postURL; // check for bad ID 

		String result = httpGet(urlPath);

		/* JSON Parser */
		JSONParser parser = new JSONParser();
		Object obj = parser.parse(result);

		JSONObject obj2 = (JSONObject) obj;
		JSONObject obj3 = (JSONObject) obj2.get("term_vectors");
		JSONObject contents = (JSONObject) obj3.get("contents");
		if (contents == null)
			return 0;
		JSONObject terms = (JSONObject) contents.get("terms");
		
		// System.out.println(token1.keySet());
		long count = terms.keySet().size();
		
		return count;
	}
	
	// calculate the TermPosting details for a term, in a given document
	public static TermPostings getTermPostings(String term, String docID)
			throws ParseException, IOException {

		/* prepare URL */
		TermPostings P = new TermPostings();

		String preURL = "http://localhost:9200/coffee5/axe_dataset/";
		String postURL = "/_termvector?pretty=true&term_statistics=true";
		String urlPath = preURL + docID + postURL; // check for bad ID 

		String result = httpGet(urlPath);

		/* JSON Parser */
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
		long count = terms.keySet().size();
		
		// System.out.println("Document length: " + count);
		
		
		/* return 0, if the term is not present in the document */
		if (token1 == null) {
			P.df = 0;
			P.tf = 0;
			P.ttf = 0;
			return P;
		}
		
		/* extract needed details */
		long tf19 = (long) token1.get("term_freq");
		long df19 = (long) token1.get("doc_freq");
		long ttf19 = (long) token1.get("ttf");
		
		/* set parameters */
		P.tf = tf19;
		P.df = df19;
		P.ttf = ttf19;
		
		/* 
		 * More details, if required. 
        JSONArray arr = (JSONArray)token1.get("tokens");
        JSONObject arrFields = (JSONObject)arr.get(0);
        System.out.println(arrFields.get("payload"));
        System.out.println(arrFields.get("position"));
        System.out.println(arrFields.get("start_offset"));
		*/
		
        /* return */
		return P;
	}

	private static Client getClient() {

		Node node = nodeBuilder().node(); // create a node
		Client client = node.client(); // create a client of that node

		return client;

	}

	public static String httpGet(String urlStr) throws IOException {
		  URL url = new URL(urlStr);
		  HttpURLConnection conn =
		      (HttpURLConnection) url.openConnection();
		  
		  // System.out.println(urlStr);
		  if (conn.getResponseCode() != 200) {
		    throw new IOException(conn.getResponseMessage());
		  }

		  // Buffer the result into a string
		  BufferedReader rd = new BufferedReader(
		      new InputStreamReader(conn.getInputStream()));
		  StringBuilder sb = new StringBuilder();
		  String line;
		  while ((line = rd.readLine()) != null) {
		    sb.append(line);
		  }
		  rd.close();

		  conn.disconnect();
		  return sb.toString();
		}
}



/* Stats for a term T found in a document D */ 
class TermPostings {
	long tf; // term frequency, term_freq (number of times the term occurs in the document)
	long df; // doc frequency, doc_freq # (the number of documents containing the current term)
	long ttf; // total term frequency, ttf # (how often a term occurs in all documents)
	
	// document details
	// sum_doc_freq, doc_count, sum_ttf # check if any such details are required. 
	
	
	public String toString() {
		return "tf: " + tf + ", " + "df: " + df + ", " + "ttf: " + ttf;
		
	}
	
}

/* Structure for storing document details, for a retrieval model */
class DocumentStats {
	String docID;
	Double score;
	
	public DocumentStats(String docID, Double score) {
		// TODO Auto-generated constructor stub
		this.docID = docID;
		this.score = score;
	}
	
	public String toString() {
		String _docID = "docID: " + docID; 
		String _score = ", score: " + score ;
		
		return _docID + _score;
		
	}
	
	
	
}


