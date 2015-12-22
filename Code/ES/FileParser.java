

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Map;

import org.elasticsearch.action.index.IndexRequestBuilder;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.node.Node;

class DocAttributes {

	private String docID;
	private String textSection;
	private int docSize;

	public String getDocID() {
		return docID;
	}

	public void setDocID(String docID) {
		this.docID = docID;
	}

	public String getTextSection() {
		return textSection;
	}

	public void setTextSection(String textSection) {
		this.textSection = textSection;
	}

	public int getDocSize() {
		return docSize;
	}

	public void setDocSize(int docSize) {
		this.docSize = docSize;
	}

}

public class FileParser {
	
	static int id = 0;
	
	
	public static String[] getStopWords() 
			throws IOException {
		// TODO Auto-generated method stub
		String pathName = "C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//stoplist.txt";
		String content = new String(Files.readAllBytes(Paths.get(pathName)));
		
		String tokens[] = content.split("\n");
		
		return tokens;
	}
	

	static String readFile(String path, Charset encoding) throws IOException {
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}
	
	
	static void indexFiles(String str, Map<String, Object> json,
			Client client, String indexName, String indexType) throws IOException {
		


		LinkedList<String> docs = new LinkedList<String>();
		docs = FileParser.fileToDocs(str);
		//System.out.println("Docs#: " + docs.size()); // remove the tags from the file

		LinkedList<DocAttributes> docList = new LinkedList<DocAttributes>();

		// String docIds = "";
		ArrayList<String> docIds = new ArrayList<String>(); 
		PrintWriter writer = new PrintWriter("C://Users//Kushagra//Desktop//CS6120_IR//Assignment//A1//AP89_DATA//AP_DATA//Cache//docIDs", "UTF-8");
		for (String d : docs) {
			docList.add(FileParser.createDocAttributes(d)); // generate JSON attributes 
		}

		@SuppressWarnings("unused")
		IndexResponse response;
		String docID;

		int count = 0;
		// String indexName = "lama";
		// String indexType = "testDocs";
		IndexRequestBuilder indexRequestBuilder;
		for (DocAttributes d : docList) {
			//System.out.println("Reached inside: ");
			
			// Prepare JSON
			json.put("docSize", d.getDocSize());
			json.put("contents", d.getTextSection());
			docID = d.getDocID().trim();
			
			id++;
			indexRequestBuilder = client.prepareIndex(indexName, indexType,
					String.valueOf(id));
				
			
			docIds.add(docID + "\n");
			
			
			
			// start the index, check in the docs iteratively
			/*
			 * response = client // (index, type, id) .prepareIndex(indexName,
			 * indexType, docID).setSource(json) .execute().actionGet();
			 */

			indexRequestBuilder.setSource(json);
			indexRequestBuilder.execute().actionGet();

			count++;
			//System.out.println("Checked in#: " + count + ", ID: " + docID);
			System.out.println(docID);
		}
		
		String S = "";
		for (String s : docIds) {
			S = S + s;
		}
		writer.println(S);
		writer.close();
	}

	static DocAttributes createDocAttributes(String doc) {
		// System.out.println("Entered createDocAttributes");
		DocAttributes d = new DocAttributes();

		// get docID
		int docID_startPos = 0;
		int docID_endPos = 0;
		String docID_startTag = "<DOCNO>";
		String docID_endTag = "</DOCNO>";

		docID_startPos = doc.indexOf(docID_startTag) + docID_startTag.length();
		docID_endPos = doc.indexOf(docID_endTag);

		String docID = doc.substring(docID_startPos, docID_endPos);
		d.setDocID(docID);

		// aggregate textSections
		String textSections = "";

		int tSec_startPos = 0;
		int tSec_endPos = 0;
		int begin = 0;
		int check = 0;

		String tSec_startTag = "<TEXT>";
		String tSec_endTag = "</TEXT>";

		while (check != -1) {
			// System.out.println(tSec_startPos + ", " + tSec_endPos);
			tSec_startPos = doc.indexOf(tSec_startTag, begin)
					+ tSec_startTag.length();
			tSec_endPos = doc.indexOf(tSec_endTag, tSec_startPos);

			textSections = textSections
					+ doc.substring(tSec_startPos, tSec_endPos);

			begin = tSec_endPos;
			check = doc.indexOf(tSec_startTag, begin);
		}

		d.setTextSection(textSections);

		// TODO: other

		// 1. docSize
		d.setDocSize(textSections.length());

		return d;
	}

	static LinkedList<String> fileToDocIDs(String str) {
		LinkedList<String> docIDs = new LinkedList<String>();

		String docID;
		String startTag = "<DOCNO>";
		String endTag = "</DOCNO>";

		// File Indexing
		int startPos = 0;
		int endPos = 0;

		// While loop details
		int check = 0;
		int begin = 0;

		// Debug
		int count = 0;
		while (check != -1) {
			startPos = str.indexOf(startTag, begin);
			endPos = str.indexOf(endTag, startPos);

			docID = str.substring(startPos + startTag.length(), endPos);
			docIDs.add(docID.trim()); // clean whitespaces
			count++;

			// while loop checks
			begin = endPos;
			check = str.indexOf(startTag, begin);

		}

		// System.out.println("Count for docIDs: " + count);
		// System.out.println("DocIDs list size: " + docIDs.size());

		return docIDs;
	}

	static LinkedList<String> fileToDocs(String str) {
		LinkedList<String> docIDs = new LinkedList<String>();

		String docID;
		String startTag = "<DOC>";
		String endTag = "</DOC>";

		// File Indexing
		int startPos = 0;
		int endPos = 0;

		// While loop details
		int check = 0;
		int begin = 0;

		// Debug
		int count = 0;
		while (check != -1) {
			startPos = str.indexOf(startTag, begin);
			endPos = str.indexOf(endTag, startPos);

			docID = str.substring(startPos + startTag.length(), endPos);
			docIDs.add(docID.trim()); // clean whitespaces
			count++;

			// while loop checks
			begin = endPos;
			check = str.indexOf(startTag, begin);

		}

		// System.out.println("Count for docs: " + count);
		// System.out.println("Docs list size: " + docIDs.size());

		return docIDs;
	}
	
	/*
	static LinkedList<String> extractTextFromFile(String str) {
		LinkedList<String> docIDs = new LinkedList<String>();

		String docID;
		String startTag = "<TEXT>";
		String endTag = "</TEXT>";

		// File Indexing
		int startPos = 0;
		int endPos = 0;

		// While loop details
		int check = 0;
		int begin = 0;

		// Debug
		int count = 0;
		while (check != -1) {
			startPos = str.indexOf(startTag, begin);
			endPos = str.indexOf(endTag, startPos);

			docID = str.substring(startPos + startTag.length(), endPos);
			docIDs.add(docID.trim()); // clean whitespaces
			count++;

			// while loop checks
			begin = endPos;
			check = str.indexOf(startTag, begin);

		}

		// System.out.println("Text section count: " + count);
		// System.out.println("Text count:" + docIDs.size());

		return docIDs;
	}
	
	*/

}

// typdef equivalent
class Document {
	String name;

	Document(String name) {
		this.name = name;
	}

	public String toString() {
		return name;
	}

}
