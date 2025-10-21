#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <locale>
#include <omp.h>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1);
    std::vector<int> R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main() {
    setlocale(LC_ALL, "rus");
    const int length = 1000;
    std::vector<int> arr(length);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < length; ++i) {
        arr[i] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < 100; ++j) {
        mergeSort(arr, 0, length - 1);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Merge Sort" << std::endl;
    std::cout << "Sorting finished" << std::endl;
    std::cout << "Time taken: " << elapsed.count() << " sec" << std::endl;

    return 0;
}
