import java.util.Comparator;

public class DocumentStatsComparator implements Comparator<DocumentStats> {

	@Override
	public int compare(DocumentStats ds1, DocumentStats ds2) {
		// TODO Auto-generated method stub

		if (ds1.score > ds2.score)
			return -1;
		else
			return 1;

	}

}
