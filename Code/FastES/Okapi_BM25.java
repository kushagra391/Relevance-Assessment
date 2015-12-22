import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class Okapi_BM25 {
	
	static double AVG_LENGTH_OF_DOCUMENTS = 177.5;
	static long TOTAL_NUMBER_OF_DOCUMENTS = 84678;
	
	
	public static ArrayList<DocumentStats> rankDocIdsForQuery_Okapi_BM25(
			String[] query, ArrayList<String> docIds) throws IOException{
		
		// Start engines 
		HashMap<String, String> map_tf = ParsePostings.getMap_TF();
		HashMap<String, String> map_df = ParsePostings.getMap_DF();
		HashMap<String, Long> 	map_len = ParsePostings.createMap_DocLength();
		
		
		ArrayList<DocumentStats> rankedDocsForQuery = new ArrayList<DocumentStats>();
		
		// initialize rankedDocsForQuery
		for (String d : docIds) {
			DocumentStats ds = new DocumentStats(d, 0.0);
			rankedDocsForQuery.add(ds);
		}
		
		for (String term : query) {
			ArrayList<DocumentStats> newRankingOfDocs = rankDocIdsForTerms_Okapi_BM25(term, docIds, query, map_tf, map_df, map_len);
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


	public static ArrayList<DocumentStats> rankDocIdsForTerms_Okapi_BM25(
			String term, ArrayList<String> docIds, String[] query, HashMap<String, String> map_tf, HashMap<String, String> map_df, HashMap<String, Long> map_len) {
		// TODO Auto-generated method stub
		
		ArrayList<DocumentStats> rankedDocForTerm = new ArrayList<DocumentStats>();
		
		for (String doc : docIds) {
			DocumentStats ds = calculateDocScoreForTerm(term, doc, query, map_tf, map_df, map_len);
			rankedDocForTerm.add(ds);
		}
		
		return rankedDocForTerm;
	}

	/* Calculate tf_wq */
	public static DocumentStats calculateDocScoreForTerm(String term,
			String docID, String[] query, HashMap<String, String> map_tf, HashMap<String, String> map_df, HashMap<String, Long> map_len) {
		// TODO Auto-generated method stub
		
		double score;
		
		long tf_wd; 
		long df_w;
		long len_d;
		double avg_len_d;
		
		
		// CONSTANTS 
		// taken from the slides
		double k1 = 1.2;
		double k2 = 500;
		double b = 0.75;
		
		long D = TOTAL_NUMBER_OF_DOCUMENTS;
		
		
		long tf_wq = getTermQueryFrequency(term, query); /* recalculate */
		
		df_w = Long.valueOf(ParsePostings.getTermPosting_df(term, docID, map_df)).longValue();
		tf_wd = Long.valueOf(ParsePostings.getTermPosting_tf(term, docID, map_tf)).longValue();
		len_d = ParsePostings.getTermLength(docID, map_len);
		avg_len_d = AVG_LENGTH_OF_DOCUMENTS; // cached result. 
		
		
		double score1;
		double score2;
		double score3;
		
		score1 = Math.log ((D + 0.5) / (df_w + 0.5));
		score2 = (tf_wd + k1* tf_wd) / (tf_wd + k1 * ((1 - b) * b * (len_d / avg_len_d) ));
		score3 = (tf_wq + k2 * tf_wq) / (tf_wq + k2);
		
		score = score1 * score2 * score3;
		
		DocumentStats ds = new DocumentStats(docID, score);
		return ds;
	}
	
	
 
	// RETURNS: the frequency of a term in a given query
	private static long getTermQueryFrequency(String term, String[] query) {
		
		long count = 0;
		
		for (String q : query){
			if (q.equals(term))
				count++;
		}
		
		return count;
	}

}
