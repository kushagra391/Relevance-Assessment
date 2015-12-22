import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class LaplaceSmoothing {
	
	static long VOCABULARY_SIZE = 17081;
	
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
		long V = VOCABULARY_SIZE;	// cached value for vocabulary size

		long tf_wd;
		long len_d;

		tf_wd = Long.valueOf(
				ParsePostings.getTermPosting_tf(term, docID, map_tf))
				.longValue();
		len_d = ParsePostings.getTermLength(docID, map_len);

		score = ((double)(tf_wd + 1))/ ((double)(len_d + V));
		score = Math.log(score);

		DocumentStats ds = new DocumentStats(docID, score);
		return ds;

	}
}
