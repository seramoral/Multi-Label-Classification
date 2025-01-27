package mulan.classifier.lazy;

import java.util.ArrayList;
import java.util.Collections;
import com.google.common.collect.MinMaxPriorityQueue;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

public class NearestNeighborsPerClass extends LinearNNSearch {

	private static final long serialVersionUID = -2704890682842160816L;

	private double[][] distancesHash;

	private double[] m_distancePreCal;

	/**
	 * It returns the k-nearest neighbors per class
	 * 
	 * @param instance
	 *            The instance
	 * @param k
	 *            The number of nearest neighbors
	 * @return An array of PriorityQueues with the k-nearest neighbors per class
	 */

	public ArrayList<Particle> kNearestNeighborsPerClassMinMaxHeaps(
			Instance instance, int k) {

		MinMaxPriorityQueue<Particle>[] kNearestNeigborsPerClass = new MinMaxPriorityQueue[m_Instances
				.numClasses()];

		for (int c = 0; c < m_Instances.numClasses(); c++) {

			kNearestNeigborsPerClass[c] = MinMaxPriorityQueue
					.orderedBy(new Particle()).maximumSize(k).create();

		}

		for (int i = 0; i < m_Instances.numInstances(); i++) {

			Instance inst = m_Instances.instance(i);

			int classValue = (int) inst.classValue();

			double distance = m_DistanceFunction.distance(instance, inst);

			kNearestNeigborsPerClass[classValue].add(new Particle(i,
					classValue, distance));

		}

		ArrayList<Particle> particles = new ArrayList<Particle>();

		for (int c = 0; c < m_Instances.numClasses(); c++) {

			while (!kNearestNeigborsPerClass[c].isEmpty()) {

				particles.add(kNearestNeigborsPerClass[c].poll());
			}
		}

		Collections.sort(particles);

		return particles;

	}

	/**
	 * It returns the k-nearest neighbors per class
	 * 
	 * @param instance
	 *            The instance
	 * @param k
	 *            The number of nearest neighbors
	 * @return An array of PriorityQueues with the k-nearest neighbors per class
	 */
	public ArrayList<NearestNeighbor> NearestNeighborsSameClass(int index) {

		MinMaxPriorityQueue<NearestNeighbor> kNearestNeigbors = MinMaxPriorityQueue
				.create().orderedBy(new NearestNeighbor())
				.maximumSize(m_Instances.numInstances() - 1).create();

		int classIndex = (int) m_Instances.instance(index).classValue();

		for (int i = 0; i < m_Instances.numInstances(); i++) {

			// skip identical instances
			if (index == i)
				continue;

			double distance = getDistance(index, i);

			kNearestNeigbors.add(new NearestNeighbor(i, distance,
					(int) m_Instances.instance(i).classValue()));

		}

		ArrayList<NearestNeighbor> nearestNeigbors = new ArrayList<NearestNeighbor>();

		//
		while (!kNearestNeigbors.isEmpty()) {

			if (kNearestNeigbors.peek().classIndex != classIndex)
				break;
			else
				nearestNeigbors.add(kNearestNeigbors.poll());

		}

		return nearestNeigbors;
	}

	public void preCalculateDistances() {

		// To store the distances. The distance among pair of instances is
		// precalculated to accelerate the computation
		distancesHash = new double[m_Instances.numInstances() - 1][];

		for (int i = 0; i < m_Instances.numInstances() - 1; i++) {

			distancesHash[i] = new double[m_Instances.numInstances() - i - 1];

			for (int j = i + 1; j < m_Instances.numInstances(); j++) {

				double temp = m_DistanceFunction.distance(
						m_Instances.instance(i), m_Instances.instance(j));

				distancesHash[i][j - i - 1] = temp;

			}
		}
	}

	// Returns the precalculate distance among i and j
	public double getDistance(int i, int j) {

		if (distancesHash == null)
			System.err.println("The distance must be precalcultated before");

		int ik = Math.min(i, j);

		int jk = Math.max(j, i);

		// System.out.println(ik+" "+jk);
		return distancesHash[ik][jk - ik - 1];
	}

	/**
	 * Returns k nearest instances in the current neighbourhood to the supplied
	 * instance, using the precal distances. This function only has sence if you
	 * want to retrieve the k nearest neighours of an instance that belongs to
	 * the training set.
	 * 
	 * @param instance
	 *            The instance
	 * @param k
	 *            The number of nearest neighbors
	 * @return An array of PriorityQueues with the k-nearest neighbors per class
	 */

	public ArrayList<Particle> kNearestNeighboursPreCalDistancesMinMaxHeap(
			int indexTarget, int k) {

		MinMaxPriorityQueue<Particle> kNearestNeigbors;

		kNearestNeigbors = MinMaxPriorityQueue.orderedBy(new Particle())
				.maximumSize(k).create();

		for (int i = 0; i < m_Instances.numInstances(); i++) {

			if (indexTarget == i)
				continue;

			double distance = getDistance(indexTarget, i);

			int classValue = (int) m_Instances.instance(i).classValue();

			kNearestNeigbors.add(new Particle(i, classValue, distance));

		}

		ArrayList<Particle> particles = new ArrayList<Particle>(k);

		while (!kNearestNeigbors.isEmpty()) {

			particles.add(kNearestNeigbors.poll());
		}

		return particles;

	}

	/**
	 * Returns k nearest instances per class in the current neighbourhood to the
	 * supplied instance, using the precal distances. This function only has
	 * sense if you want to retrieve the k nearest neighours of an instance that
	 * belongs to the training set. It adjusts the number of neighbours taking
	 * into account the minimal number of neighbors that can be retrieved.
	 * 
	 * The nearest neighbours per class are retrieved in order of instances
	 * 
	 * @param instance
	 *            The instance
	 * @param k
	 *            The number of nearest neighbors
	 * @return An array of PriorityQueues with the k-nearest neighbors per class
	 */

	public ArrayList<Particle> kNearestNeighboursPreCalDistancesBinaryClassMinMaxHeap(
			int indexTarget, int k) {

		MinMaxPriorityQueue<Particle>[] kNearestNeigborsPerClass = new MinMaxPriorityQueue[2];

		kNearestNeigborsPerClass[0] = MinMaxPriorityQueue
				.orderedBy(new Particle()).maximumSize(k).create();

		kNearestNeigborsPerClass[1] = MinMaxPriorityQueue
				.orderedBy(new Particle()).maximumSize(k).create();

		for (int i = 0; i < m_Instances.numInstances(); i++) {

			if (indexTarget == i)
				continue;

			Instance inst = m_Instances.instance(i);

			int classValue = (int) inst.classValue();

			double distance = getDistance(i, indexTarget);

			kNearestNeigborsPerClass[classValue].add(new Particle(i,
					classValue, distance));

		}

		ArrayList<Particle> particles = new ArrayList<Particle>();

		for (int c = 0; c < m_Instances.numClasses(); c++) {

			while (!kNearestNeigborsPerClass[c].isEmpty()) {

				particles.add(kNearestNeigborsPerClass[c].poll());
			}
		}

		Collections.sort(particles);

		return particles;
	}

	/**
	 * Returns k nearest instances per class in the current neighbourhood to the
	 * supplied instance, using the precal distances. This function only has
	 * sense if you want to retrieve the k nearest neighours of an instance that
	 * belongs to the training set. It adjusts the number of neighbours taking
	 * into account the minimal number of neighbors that can be retrieved.
	 * 
	 * The nearest neighbours per class are retrieved in order of instances
	 * 
	 * @param instance
	 *            The instance
	 * @param k
	 *            The number of nearest neighbors
	 * @return An array of PriorityQueues with the k-nearest neighbors per class
	 */

	public ArrayList<Object>[] kNearestNeighboursMultiClass(int indexTarget,
			int k) {

		// all the instances are returned according their distances
		Instances NN = null;

		try {
			
			NN = kNearestNeighbours(m_Instances.instance(indexTarget),
					m_Instances.numInstances());

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Empty set
		ArrayList<Object> kNNPerClassIntances = new ArrayList<Object>(k
				* m_Instances.numClasses());

		ArrayList<Object> distanceskNNPerClassIntances = new ArrayList<Object>(
				k * m_Instances.numClasses());

		ArrayList<Object> kNN = new ArrayList<Object>(k);

		ArrayList<Object> distanceskNN = new ArrayList<Object>(k);

		int[] kNNPerClass = new int[m_Instances.numClasses()];

		int index = 0;

		// Each class must be there k neighbors
		while (Utils.sum(kNNPerClass) < k * m_Instances.numClasses()) {
			
			if(index==NN.size())
			{
				break;
				
			}
			
			int classIndex = (int) NN.instance(index).classValue();

			if (kNNPerClass[classIndex] < k) {

				kNNPerClass[classIndex]++;

				kNNPerClassIntances.add(NN.instance(index));

				distanceskNNPerClassIntances.add(m_Distances[index]);
			}

			if (kNN.size() < k) {

				kNN.add(NN.instance(index));
				distanceskNN.add(m_Distances[index]);
			}

			index++;
		}

		ArrayList<Object>[] arr = new ArrayList[4];

		arr[0] = kNNPerClassIntances;

		arr[1] = distanceskNNPerClassIntances;

		arr[2] = kNN;

		arr[3] = distanceskNN;

		return arr;
	}

	public double[] getM_distancePreCal() {
		return m_distancePreCal;
	}

	public void setM_distancePreCal(double[] m_distancePreCal) {
		this.m_distancePreCal = m_distancePreCal;
	}

	/**
	 * Returns k nearest instances in the current neighbourhood to the supplied
	 * instance.
	 * 
	 * @param target
	 *            The instance to find the k nearest neighbours for.
	 * @param kNN
	 *            The number of nearest neighbours to find.
	 * @return the k nearest neighbors
	 * @throws Exception
	 *             if the neighbours could not be found.
	 */
	public int[] kNearestNeighboursIndices(Instance target, int kNN)
			throws Exception {

		// debug
		boolean print = false;

		if (m_Stats != null)
			m_Stats.searchStart();

		MyHeap heap = new MyHeap(kNN);
		double distance;
		int firstkNN = 0;
		for (int i = 0; i < m_Instances.numInstances(); i++) {
			if (target == m_Instances.instance(i)) // for hold-one-out
													// cross-validation
				continue;
			if (m_Stats != null)
				m_Stats.incrPointCount();
			if (firstkNN < kNN) {
				if (print)
					System.out.println("K(a): "
							+ (heap.size() + heap.noOfKthNearest()));
				distance = m_DistanceFunction.distance(target,
						m_Instances.instance(i), Double.POSITIVE_INFINITY,
						m_Stats);
				if (distance == 0.0 && m_SkipIdentical)
					if (i < m_Instances.numInstances() - 1)
						continue;
					else
						heap.put(i, distance);
				heap.put(i, distance);
				firstkNN++;
			} else {
				MyHeapElement temp = heap.peek();
				if (print)
					System.out.println("K(b): "
							+ (heap.size() + heap.noOfKthNearest()));
				distance = m_DistanceFunction.distance(target,
						m_Instances.instance(i), temp.distance, m_Stats);
				if (distance == 0.0 && m_SkipIdentical)
					continue;
				if (distance < temp.distance) {
					heap.putBySubstitute(i, distance);
				} else if (distance == temp.distance) {
					heap.putKthNearest(i, distance);
				}

			}
		}

		m_Distances = new double[heap.size() + heap.noOfKthNearest()];
		int[] indices = new int[heap.size() + heap.noOfKthNearest()];
		int i = 1;
		MyHeapElement h;
		while (heap.noOfKthNearest() > 0) {
			h = heap.getKthNearest();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = h.distance;
			i++;
		}
		while (heap.size() > 0) {
			h = heap.get();
			indices[indices.length - i] = h.index;
			m_Distances[indices.length - i] = h.distance;
			i++;
		}

		m_DistanceFunction.postProcessDistances(m_Distances);

		if (m_Stats != null)
			m_Stats.searchFinish();

		return indices;
	}
}