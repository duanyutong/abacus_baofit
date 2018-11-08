/*
  -- changes by Ryuichiro ---

  l.57 memcpy(a, arr2...  >>  memcpy(a, arr1...
*/

typedef Galaxy Merge;
STimer Sorting, Merging;

void merge(Merge a[], Merge temp[], int size, int size2) {
    // Adapted from
    // https://github.com/serendependy/parallel-j/blob/master/OpenMP/mergeSort-omp.c
    // size2 is the division between the two sublists
    // We are actually moving the data from a[] to temp[].
    int i1 = 0;
    int i2 = size2;
    int it = 0;
    while(i1 < size2 && i2 < size) {
	if (a[i1] < a[i2]) temp[it++] = a[i1++];
	    else temp[it++] = a[i2++];
    }
    while (i1 < size2) temp[it++] = a[i1++];
    while (i2 < size) temp[it++] = a[i2++];
    return;
}

void mergesort_parallel_omp(Merge a[], const int size, Merge temp[], const int threads) {
    // We need to divide the list into threads pieces.
    // Define the boundaries: sect[j]..sect[j+1]-1, inclusive
    int sect[threads+1];
    for (int j=0; j<threads; j++) sect[j]=j*size/threads; sect[threads]=size;
    Sorting.Start();
#pragma omp parallel for schedule(static,1)
    for (int j=0; j<threads; j++) std::sort(a+sect[j], a+sect[j+1]);
    Sorting.Stop();
    // Now we have the sorted sections; we need to merge them.
    // Eventually this might be parallelized too, but let's get something working
    Merging.Start();
    Merge *arr1=a, *arr2=temp;
    for (int m=2; m<threads*2; m*=2) {
        // Merge in blocks of m.
#pragma omp parallel for schedule(static,1)
	for (int j=0; j<threads; j+=m) {
	    // Block goes from j to j+m/2 and then j+m/2 to j+m, if these exist
	    int s1 = j;
	    int s2 = j+m/2;
	    int s3 = j+m;
	    if (s2>=threads) continue;   // No merging is needed.
	    if (s3>=threads) s3=threads;  // Truncate this region
	    // printf("Merging %d-%d to %d-%d\n", s1, s2-1, s2, s3-1);
	    merge(arr1+sect[s1], arr2+sect[s1], sect[s3]-sect[s1], sect[s2]-sect[s1]);
	}
	// Better sorted data is now in arr2, so swap the names
	std::swap(arr1,arr2);
    }
    // If we ended up with an odd number of passes, we have to flip back
    if (arr2==a) memcpy(a, arr1, sizeof(Merge)*size);
    Merging.Stop();
    return;
}
