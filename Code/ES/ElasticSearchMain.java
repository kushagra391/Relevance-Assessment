import org.elasticsearch.action.admin.indices.create.CreateIndexRequestBuilder;
import org.elasticsearch.client.Client;







import static org.elasticsearch.common.xcontent.XContentFactory.jsonBuilder;
import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

// import RetrievalA1.FileParser;



import org.elasticsearch.action.admin.indices.create.CreateIndexRequestBuilder;
import org.elasticsearch.action.admin.indices.delete.DeleteIndexRequestBuilder;
import org.elasticsearch.action.admin.indices.exists.indices.IndicesExistsResponse;
import org.elasticsearch.action.admin.indices.mapping.put.PutMappingResponse;
import org.elasticsearch.action.get.GetRequestBuilder;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.index.IndexRequestBuilder;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.search.SearchType;
import org.elasticsearch.action.termvector.TermVectorRequestBuilder;
import org.elasticsearch.action.termvector.TermVectorResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.common.settings.ImmutableSettings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.index.query.MoreLikeThisQueryParser.Fields;
import org.elasticsearch.node.Node;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

public class ElasticSearchMain {
	// global statics
	final static String indexName = "coffee5";
	final static String documentType = "axe_dataset";
	final static String fieldName = "contents";
	
	static String dir = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//ap89_collection";
	
	public static void main(String[] args) 
			throws IOException, InterruptedException, FileNotFoundException {

		// create client
		final Client client = getClient();
		
		
		// placeholder for pre-index operations
		final CreateIndexRequestBuilder createIndexRequestBuilder = client
				.admin().indices().prepareCreate(indexName);
		
		
		/* --- Add mapping and Settings have been added to the index --- */ 
		
		// Add Mapping 
		XContentBuilder mappingBuilder = createMapper();
		createIndexRequestBuilder.addMapping(documentType, mappingBuilder); // mapping added
		
		// Add Setting
		XContentBuilder settingsBuilder = createSetting();
		createIndexRequestBuilder.setSettings(settingsBuilder); 			// setting added
		
		createIndexRequestBuilder.execute().actionGet(); 					// CreateIndexRequestBuilder executed
		
		/* --- Mapping and Settings have been added to the index  --- */ 
		
		
				
		// parse from path. Use for now 
		final String documentId = "1";
		final String value = "This is a test document. First Entry.";
		final IndexRequestBuilder indexRequestBuilder = client.prepareIndex(
				indexName, documentType, documentId);
		
		/* Add one test document */
		
		Map<String, Object> json = new HashMap<String, Object>();
		json.put(fieldName, value);
		
		indexRequestBuilder.setSource(json);	
		indexRequestBuilder.execute().actionGet();
		
				
		
		// parse documents
		

		Map<String, Object> JSON = new HashMap<String, Object>(); // placeholder for json string
		// Node node = nodeBuilder().node(); // create a node
		
		// Parse through the directories and index all documents
		Files.walk(Paths.get(dir)).forEach(filePath -> {
			if (Files.isRegularFile(filePath)) {
				try {
					String doc = new String(Files.readAllBytes(filePath));
					FileParser.indexFiles(doc, JSON, client, indexName, documentType);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
		
		
		
		// 
		
		TermVectorRequestBuilder resp = getClient()
				.prepareTermVector(indexName, documentType, documentId)
				.setPayloads(true).setOffsets(true).setPositions(true)
				.setSelectedFields();
		TermVectorResponse response = resp.execute().actionGet();
		
		System.out.println("size: " + response);
		System.out.println("Number of terms: " + response.getFields().size());
		System.out.println("Number of terms: " + response.getFields().terms(fieldName).size());
		System.out.println("Number of terms: " + response.getFields().terms(fieldName).getDocCount());
		
		
		
	
		
		client.close(); // close the client
		

		// parse queries
		
		String Q = "Document will discuss allegations, or measures being taken against, corrupt public officials of any governmental jurisdiction worldwide.";
		String q[] = Q.split(" |\\.|\\,");
		

		// query processing
		
		/*
		 * It will be helpful if you write a method which takes a term as a
		 * parameter and retrieves the postings for that term from
		 * elasticsearch.
		 * 
		 */

		// query results

		//

		// query score for every documents

		// scoring models

		/*
		 *  ** VARIABLES ** 
		 *  -- document length - EASY 
		 *  -- average length of document - EASY 
		 *  -- total number of documents - EASY 
		 *  
		 *  -- tf(w,q) term frequency of term w in query q - EASY 
		 *  -- c(f), d(f), t(f) -- from ES API
		 *  -- c(f), d(f), t(f) -- from ES API 
		 *  
		 *  -- * vocabulary list - MED 
		 */
		
		/*
		 * ** INFRA ** 
		 * -- trec_eval
		 * -- query result
		 * -- document parser from path
		 */
		
		
	}

	private static XContentBuilder createSetting() throws IOException {
		
		String stops[] = FileParser.getStopWords();
		XContentBuilder settingsBuilder = jsonBuilder()
                .startObject()
                .startObject("analysis")
                    .startObject("analyzer")
                        .startObject("fulltext_analyzer")
                            .field("type", "standard")
                            .field("tokenizer", "whitespace")
                            .field("filter", new String[]{"type_as_payload", "lowercase"})
                            .field("stopwords", stops)
                        .endObject()
                    .endObject()
                .endObject();
		
		return settingsBuilder;
		
	}

	private static XContentBuilder createMapper() throws IOException {
		
		XContentBuilder mappingBuilder = jsonBuilder()
                .startObject()
                     .startObject(documentType)
                          .startObject("properties")
                              .startObject("contents")
                                  .field("type", "string")
                                  .field("index_analyzer", "fulltext_analyzer")
                                  .field("term_vector", "with_positions_offsets_payloads")
                                  .field("store", true)
                               .endObject()
                          .endObject()
                      .endObject()
                   .endObject();
		
		return mappingBuilder;
	}

	private static Client getClient() {

		Node node = nodeBuilder().node(); // create a node
		Client client = node.client(); // create a client of that node

		return client;

	}

}
