package mulan.classifier.lazy;

import java.util.Comparator;

public class NearestNeighbor implements Comparable<NearestNeighbor>,
		Comparator<NearestNeighbor> {

	// The index of the instance
	public int index;

	// The distance
	public double distance;

	// The class of the instance
	public int classIndex;

	public NearestNeighbor() {
	}

	public NearestNeighbor(int index, double distance, int classIndex) {
		this.index = index;
		this.distance = distance;
		this.classIndex = classIndex;
	}

	@Override
	public int compareTo(NearestNeighbor o) {

		if (distance < o.distance)
			return -1;
		else
		{
			if(distance==o.distance)
			{
				if(classIndex!=o.classIndex)
					return -1;
				else
					return 0;
			}
			else
				return 1;
		}

	}

	@Override
	public String toString() {
		return "index: " + index + " " + "distance: " + distance;
	}

	public boolean equals(Object o) {

		NearestNeighbor n = (NearestNeighbor) o;

		return (index == n.index && distance == n.distance);
	}

	@Override
	public int compare(NearestNeighbor o1, NearestNeighbor o2) {

		return o1.compareTo(o2);
	}

}
