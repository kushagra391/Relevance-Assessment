import java.io.IOException;
import java.util.ArrayList;

import org.json.simple.parser.ParseException;


public class Okapi_BM25 {
	public static ArrayList<DocumentStats> rankDocIdsForQuery_Okapi_BM25(
			String[] query, ArrayList<String> docIds) {
		
		ArrayList<DocumentStats> rankedDocsForQuery = new ArrayList<DocumentStats>();
		
		// initialize rankedDocsForQuery
		for (String d : docIds) {
			DocumentStats ds = new DocumentStats(d, 0.0);
			rankedDocsForQuery.add(ds);
		}
		
		for (String term : query) {
			ArrayList<DocumentStats> newRankingOfDocs = rankDocIdsForTerms_Okapi_BM25(term, docIds, query);
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
			String term, ArrayList<String> docIds, String[] query) {
		// TODO Auto-generated method stub
		
		ArrayList<DocumentStats> rankedDocForTerm = new ArrayList<DocumentStats>();
		
		for (String doc : docIds) {
			DocumentStats ds = calculateDocScoreForTerm(term, doc, query);
			rankedDocForTerm.add(ds);
		}
		
		return rankedDocForTerm;
	}

	/* Calculate tf_wq */
	public static DocumentStats calculateDocScoreForTerm(String term,
			String docID, String[] query) {
		// TODO Auto-generated method stub
		
		double score;
		
		long tf_wd; 
		long df_w;
		long len_d;
		double avg_len_d;
		
		
		// CONSTANTS
		double k1 = 1.2;
		double k2 = 500;
		double b = 0.75;
		
		long D = 80000;
		
		
		long tf_wq = 1; /* recalculate */
		
		tf_wd = TermVectorStats.getTermPostings(term, docID).tf;
		df_w = TermVectorStats.getTermPostings(term, docID).df;
		len_d = TermVectorStats.getDocLength(docID);
		avg_len_d = len_d; // for now, cache the result for later use. 
		
		
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
	
	
	long get_df_wq(String term, String[] query) {
		
		long count = 0;
		
		for (String q : query){
			if (q.equals(term))
				count++;
		}
		
		return count;
	}

}
