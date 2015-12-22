import java.io.BufferedReader;
import java.io.FileReader;


public class QrelParser {
	
	private static String qrelPath = "C://Users//Kushagra//Desktop//CS6120_IR//HW_5//data//qrels.adhoc.51-100.AP89.txt";
	private static int QREL_QUERYID_LOCATION = 0;
	private static int QREL_DOCID_LOCATION = 2;
	private static int QREL_ASSESSMENT_LOCATION = 3;

	public static void main(String[] main) {
		
		try (BufferedReader br = new BufferedReader(new FileReader(qrelPath))) {
			String line;
			while ((line = br.readLine()) != null) {
				// process the line.
				String[] tokens = line.split(" ");

				String _queryID = tokens[QREL_QUERYID_LOCATION];
				int queryID = Integer.parseInt(_queryID);
				if (!FeaturesParser.trainingSet.contains(queryID)) // skip !trainingSet
					continue;

				String _rating = tokens[QREL_ASSESSMENT_LOCATION];
				int rating = Integer.parseInt(_rating);
				String docID = tokens[QREL_DOCID_LOCATION];

			}
		}
		
		
	}
	
	

}
