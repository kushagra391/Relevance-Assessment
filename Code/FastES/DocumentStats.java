import org.w3c.dom.views.DocumentView;


public class DocumentStats {
	String docID;
	Double score;
	
	public DocumentStats(String docID, Double score) {
		// TODO Auto-generated constructor stub
		this.docID = docID;
		this.score = score;
	}
	
	
	public String toString(){
		return "docID: " + docID + ", score: " + score;
	}

}
