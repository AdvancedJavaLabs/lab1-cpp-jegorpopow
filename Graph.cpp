#include "Graph.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>

Graph::Graph(int vertices) : V(vertices), adjList(vertices) {}

void Graph::addEdge(int src, int dest) {
  if (src < 0 || dest < 0 || src >= V || dest >= V)
    return;
  auto &vec = adjList[src];
  if (std::find(vec.begin(), vec.end(), dest) == vec.end()) {
    vec.push_back(dest);
  }
}

void Graph::parallelBFS(int startVertex) {
  std::vector<std::atomic<bool>> visited(V);

  std::vector<int> level(V, 0);
  std::vector<int> next_level(V, 0);
  level[0] = startVertex;
  std::size_t level_size = 1;
  std::atomic<std::size_t> next_level_size = 0;

  const std::size_t THREAD_NUMBER = std::thread::hardware_concurrency() - 1;

  auto job = [&](std::size_t tread_idx) {
    for (std::size_t i = tread_idx; i < level_size; i += THREAD_NUMBER) {
      int u = level[i];
      for (int n : adjList[u]) {
        bool old = false;
        if (visited[n].compare_exchange_strong(old, true)) {
          std::size_t index =
              next_level_size.fetch_add(1, std::memory_order::relaxed);
          next_level[index] = n;
        }
      }
    }
  };

  while (level_size != 0) {
    std::vector<std::thread> pool;
    pool.reserve(THREAD_NUMBER);
    for (std::size_t i = 0; i < THREAD_NUMBER; i++) {
      pool.emplace_back(job, i);
    }

    for (auto &thread : pool) {
      thread.join();
    }

    level_size = next_level_size;
    std::swap(level, next_level);
    next_level_size = 0;
  }

  for (std::size_t i = 0; i < V; i++) {
    if (!visited[i]) {
      std::cout << "visited[" << i << "] = false\n";
    }
  }
}

void Graph::bfs(int startVertex) {
  if (startVertex < 0 || startVertex >= V)
    return;
  std::vector<char> visited(V, 0);
  std::queue<int> q;

  visited[startVertex] = 1;
  q.push(startVertex);

  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int n : adjList[u]) {
      if (!visited[n]) {
        visited[n] = 1;
        q.push(n);
      }
    }
  }
}

int Graph::vertices() const { return V; }
