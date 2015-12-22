import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class QueryRetriever {

	static String indexName = "coffee5";
	static String indexType = "axe_dataset";

	public static void main(String[] args) throws IOException, ParseException {

		String Q = "Document will identify acquisition by the U.S. Army of specified advanced weapons systems";
		String query[] = Q.split(" ");
		ArrayList<String> docList = getDocListsFromQuery(query);

		System.out.println(docList.size());

	}

	// add checks for null cases when the word is a stopword / not present in
	// the corpus.
	public static ArrayList<String> retrieveDocIDs(String json)
			throws ParseException {

		ArrayList<String> docIDs = new ArrayList<String>();

		JSONParser parser = new JSONParser();
		Object parsedResult = parser.parse(json);
		JSONObject result = (JSONObject) parsedResult;
		JSONObject hits = (JSONObject) result.get("hits");
		JSONArray InnerHits = (JSONArray) hits.get("hits");

		for (int i = 0; i < InnerHits.size(); i++) {
			JSONObject doc1 = (JSONObject) InnerHits.get(i);
			String docID = (String) doc1.get("_id");
			docIDs.add(docID);
		}

		return docIDs;
	}

	public static ArrayList<String> getDocListsFromQuery(String[] query)
			throws IOException, ParseException {
		// TODO Auto-generated method stub

		ArrayList<String> docLists = new ArrayList<String>();

		int SIZE = 50000; // max possible df

		String preURL = "http://localhost:9200/coffee5/_search?q=contents:";

		HashSet<String> docIDs = new HashSet<String>();

		for (String q : query) {
			if (q.equals(""))
				continue;
			if (q.equals(" "))
				continue;
			
			String urlPath = preURL + q + "&size=" + SIZE;
			String result = TermVectorStats.httpGet(urlPath);

			ArrayList<String> resultDocList = retrieveDocIDs(result);

			docIDs.addAll(resultDocList);
		}

		docLists.addAll(docIDs);

		return docLists;
	}

	/*
	 * private static ArrayList<String> getDocListsFromQuery(String[] query)
	 * throws IOException { // TODO Auto-generated method stub
	 * 
	 * String urlJSON; String urlPath;
	 * 
	 * urlPath = "http://localhost:9200/coffee5/_search?"; urlJSON =
	 * createJSONForQuery(query);
	 * 
	 * String url = urlPath + urlJSON; String result =
	 * TermVectorStats.httpGet(url);
	 * 
	 * ArrayList<String> docList = extractDocIDs(result);
	 * 
	 * 
	 * return docList; }
	 * 
	 * private static ArrayList<String> extractDocIDs(String result) { // TODO
	 * Auto-generated method stub return null; }
	 * 
	 * private static String createJSONForQuery(String[] query) { // TODO
	 * Auto-generated method stub
	 * 
	 * 
	 * 
	 * return null; }
	 */

}
