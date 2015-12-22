import java.io.IOException;
import java.util.ArrayList;

import org.json.simple.parser.ParseException;


public class OkapiTF {
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
		
		tf_wd = TermVectorStats.getTermPostings(term, docID).tf;
		len_d = TermVectorStats.getDocLength(docID);
		avg_len_d = len_d; // for now, cache the result for later use. 
		
		score = tf_wd / (tf_wd + 0.5 + 1.5 * (len_d / avg_len_d));
		
		DocumentStats ds = new DocumentStats(docID, score);
		return ds;
	}

	
	
}
