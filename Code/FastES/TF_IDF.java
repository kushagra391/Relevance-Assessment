import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class TF_IDF {
	static double AVG_LENGTH_OF_DOCUMENTS = 177.5;
	static long TOTAL_NUMBER_OF_DOCUMENTS = 84678;

	public static ArrayList<DocumentStats> rankDocIdsForQuery_TF_IDF(
			String[] query, ArrayList<String> docIds) throws IOException {

		// Start engines
		HashMap<String, String> map_tf = ParsePostings.getMap_TF();
		HashMap<String, String> map_df = ParsePostings.getMap_DF();
		HashMap<String, Long> map_len = ParsePostings.createMap_DocLength();

		ArrayList<DocumentStats> rankedDocsForQuery = new ArrayList<DocumentStats>();

		// initialize rankedDocsForQuery
		for (String d : docIds) {
			DocumentStats ds = new DocumentStats(d, 0.0);
			rankedDocsForQuery.add(ds);
		}

		for (String term : query) {
			ArrayList<DocumentStats> newRankingOfDocs = rankDocIdsForTerms_TF_IDF(
					term, docIds, map_tf, map_df, map_len);
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

	public static ArrayList<DocumentStats> rankDocIdsForTerms_TF_IDF(
			String term, ArrayList<String> docIds,
			HashMap<String, String> map_tf, HashMap<String, String> map_df,
			HashMap<String, Long> map_len) {
		// TODO Auto-generated method stub

		ArrayList<DocumentStats> rankedDocForTerm = new ArrayList<DocumentStats>();

		for (String doc : docIds) {
			DocumentStats ds = calculateDocScoreForTerm(term, doc, map_tf,
					map_df, map_len);
			rankedDocForTerm.add(ds);
		}

		return rankedDocForTerm;
	}

	// Cache D
	public static DocumentStats calculateDocScoreForTerm(String term,
			String docID, HashMap<String, String> map_tf,
			HashMap<String, String> map_df, HashMap<String, Long> map_len){
		// TODO Auto-generated method stub

		double score;

		long tf_wd;
		long len_d;
		long df_w;
		double avg_len_d;

		double D = TOTAL_NUMBER_OF_DOCUMENTS;

		df_w = Long.valueOf(ParsePostings.getTermPosting_df(term, docID, map_df)).longValue();
		tf_wd = Long.valueOf(ParsePostings.getTermPosting_tf(term, docID, map_tf)).longValue();
		len_d = ParsePostings.getTermLength(docID, map_len);
		avg_len_d = AVG_LENGTH_OF_DOCUMENTS; // Cached

		score = tf_wd / (tf_wd + 0.5 + 1.5 * (len_d / avg_len_d));
		score = Math.log((double) D / df_w);

		if (df_w == 0)
			score = 0;

		DocumentStats ds = new DocumentStats(docID, score);
		return ds;
	}

}
