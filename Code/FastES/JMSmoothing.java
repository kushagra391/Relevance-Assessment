import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class JMSmoothing {
	public static ArrayList<DocumentStats> rankDocIdsForQuery_Okapi(
			String[] query, ArrayList<String> docIds) throws IOException {

		// Generate Map Engine
		HashMap<String, String> map_tf = ParsePostings.getMap_TF();
		HashMap<String, Long> 	map_len = ParsePostings.createMap_DocLength();

		ArrayList<DocumentStats> rankedDocsForQuery = new ArrayList<DocumentStats>();

		// initialize rankedDocsForQuery
		for (String d : docIds) {
			DocumentStats ds = new DocumentStats(d, 0.0);
			rankedDocsForQuery.add(ds);
		}

		for (String term : query) {
			ArrayList<DocumentStats> newRankingOfDocs = rankDocIdsForTerms_Okapi(
					term, docIds, map_tf, map_len);
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

		for (int i = 0; i < size; i++) {

			if (!rankedDocsForQuery.get(i).docID
					.equals(newRankingOfDocs.get(i).docID))
				System.out.println("FATAL ERROR");

			rankedDocsForQuery.get(i).score = rankedDocsForQuery.get(i).score
					+ newRankingOfDocs.get(i).score;
		}

	}

	public static ArrayList<DocumentStats> rankDocIdsForTerms_Okapi(
			String term, ArrayList<String> docIds,
			HashMap<String, String> map_tf, HashMap<String, Long> map_len) {
		// TODO Auto-generated method stub

		ArrayList<DocumentStats> rankedDocForTerm = new ArrayList<DocumentStats>();

		for (String doc : docIds) {
			DocumentStats ds = calculateDocScoreForTerm(term, doc, map_tf, map_len);
			rankedDocForTerm.add(ds);
		}

		return rankedDocForTerm;
	}

	public static DocumentStats calculateDocScoreForTerm(String term,
			String docID, HashMap<String, String> map_tf, HashMap<String, Long> map_len) {

		System.out.println("term: " + term);
		System.out.println("docID: " + docID);

		double score;
		double score1;
		double score2;
		
		double lambda = 0.75;

		long tf_wd;
		long len_d;

		tf_wd = Long.valueOf(
				ParsePostings.getTermPosting_tf(term, docID, map_tf))
				.longValue();
		len_d = ParsePostings.getTermLength(docID, map_len);
		
		score1 = lambda * (tf_wd / len_d);
		score2 = (1-lambda) * ((double)residual_tf_wd(term, docID, map_tf) / (double)residual_len_d(docID, map_len));
		score = score1 + score2;
		
		DocumentStats ds = new DocumentStats(docID, score);
		return ds;

	}

	private static long residual_len_d(String docID, HashMap<String, Long> map_len) {
		// TODO Auto-generated method stub
		
		long sigmaSum = 0;
		
		for (String key : map_len.keySet()) {
			if (!key.equals(docID)){
				sigmaSum = sigmaSum + map_len.get(key);
			} 
		}
		
		return sigmaSum;
	}

	private static long residual_tf_wd(String term, String docID, HashMap<String, String> map_tf) {
		// TODO Auto-generated method stub
		
		long sigmaSum = 0;
		
		for (String key : map_tf.keySet()) {
			if (!key.equals(term)) {
				sigmaSum = sigmaSum + (Long.valueOf(
						ParsePostings.getTermPosting_tf(term, docID, map_tf))
						.longValue());
			}
		}
		
		
		return sigmaSum;
	}
}
