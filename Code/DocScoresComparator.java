import java.util.Comparator;


public class DocScoresComparator implements Comparator<DocScore> {

	@Override
	public int compare(DocScore ds1, DocScore ds2) {

		if (ds1.score > ds2.score)
			return -1;
		else
			return 1;

	}
}
