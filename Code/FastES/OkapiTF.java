import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

// import org.json.simple.parser.ParseException;


public class OkapiTF {
	public static ArrayList<DocumentStats> rankDocIdsForQuery_Okapi(
			String[] query, ArrayList<String> docIds) throws IOException {
		
		// Generate Map Engine
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
			ArrayList<DocumentStats> newRankingOfDocs = rankDocIdsForTerms_Okapi(term, docIds, map_tf);
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
			String term, ArrayList<String> docIds, HashMap<String, String> map_tf) {
		// TODO Auto-generated method stub
		
		ArrayList<DocumentStats> rankedDocForTerm = new ArrayList<DocumentStats>();
		
		for (String doc : docIds) {
			DocumentStats ds = calculateDocScoreForTerm(term, doc, map_tf);
			rankedDocForTerm.add(ds);
		}
		
		return rankedDocForTerm;
	}


	public static DocumentStats calculateDocScoreForTerm(String term,
			String docID, HashMap<String, String> map_tf) {
		// TODO Auto-generated method stub
		
		System.out.println("term: " + term);
		System.out.println("docID: " + docID);
		
		// DocumentStats temp = new DocumentStats(docID, (double) 0);
		// return temp;
		
		
		double score;
		
		long tf_wd; 
		long len_d;
		double avg_len_d;
		
		tf_wd = Long.valueOf(ParsePostings.getTermPosting_tf(term, docID, map_tf)).longValue();
		len_d = 177; // TermVectorStats.getDocLength(docID);
		avg_len_d = len_d; // for now, cache the result for later use. 
		
		score = tf_wd / (tf_wd + 0.5 + 1.5 * (len_d / avg_len_d));
		
		DocumentStats ds = new DocumentStats(docID, score);
		return ds;
		
	}

	
	
}
