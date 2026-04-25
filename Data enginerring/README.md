# 🚀 Data Engineering — Zero to Hero

> A concise, structured guide covering the foundational concepts of modern Data Engineering — from raw data to real-time pipelines.

![Data Engineering](https://img.shields.io/badge/Data%20Engineering-Fundamentals-blue?style=for-the-badge&logo=apache-spark&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📚 Table of Contents

- [What is Data?](#-what-is-data)
- [What is Data Engineering?](#-what-is-data-engineering)
- [Roles of a Data Engineer](#-roles-of-a-data-engineer)
- [Types of Databases](#-types-of-databases)
- [OLTP Databases](#-oltp-databases)
- [SQL](#-sql)
- [Hadoop & HDFS](#-hadoop--hdfs)
- [MapReduce](#-mapreduce)
- [Apache Spark](#-apache-spark)
- [Apache Flink](#-apache-flink)
- [Apache Kafka & Stream Processing](#-apache-kafka--stream-processing)

---

## 🔷 What is Data?

**Data** is raw, unprocessed facts and figures collected from various sources. It can be:

| Type | Description | Example |
|------|-------------|---------|
| **Structured** | Organized in rows/columns | SQL tables, CSV files |
| **Semi-structured** | Partially organized | JSON, XML, Logs |
| **Unstructured** | No fixed format | Images, Audio, Video, PDFs |

> 💡 **Key Insight:** Data by itself has no meaning. It becomes **information** when processed, and **knowledge** when applied.

---

## 🔷 What is Data Engineering?

**Data Engineering** is the practice of designing, building, and maintaining the infrastructure and pipelines that enable data collection, storage, transformation, and delivery — making raw data usable for analysts, scientists, and decision-makers.

```
Raw Data  →  Ingestion  →  Storage  →  Processing  →  Serving  →  Insights
```

---

## 🔷 Roles of a Data Engineer

### 1️⃣ Pipeline Builder
Designs and maintains **ETL/ELT pipelines** (Extract, Transform, Load) that move data from source systems to storage/analytics layers.

### 2️⃣ Data Architect
Defines the **data architecture** — how data flows, where it's stored, and what systems interact with each other (data lakes, warehouses, lakehouses).

### 3️⃣ Infrastructure Manager
Manages and optimizes **data infrastructure** — cloud storage (S3, GCS), compute (Spark clusters), and orchestration tools (Airflow, Prefect).

### 4️⃣ Collaborator & Enabler
Works closely with **Data Scientists, Analysts, and ML Engineers** to ensure data is clean, accessible, and reliable for downstream use cases.

---

## 🔷 Types of Databases

| Type | Full Form | Use Case | Examples |
|------|-----------|----------|---------|
| **Relational** | RDBMS | Structured data, transactions | PostgreSQL, MySQL, Oracle |
| **NoSQL** | Not Only SQL | Flexible schemas, scale | MongoDB, Cassandra, DynamoDB |
| **Time-Series** | TSDB | Metrics, IoT, logs | InfluxDB, TimescaleDB |
| **Graph** | Graph DB | Relationships, networks | Neo4j, Amazon Neptune |
| **In-Memory** | Cache DB | Ultra-fast lookups | Redis, Memcached |
| **Columnar / OLAP** | Analytical DB | Big data analytics | Snowflake, BigQuery, Redshift |

---

## 🔷 OLTP Databases

**OLTP** = *Online Transaction Processing*

OLTP databases are optimized for **high-speed, short transactions** — the backbone of day-to-day business operations.

### Key Characteristics:
- ✅ High volume of small read/write operations
- ✅ ACID compliance (Atomicity, Consistency, Isolation, Durability)
- ✅ Normalized schema to reduce redundancy
- ✅ Optimized for **INSERT / UPDATE / DELETE**

### OLTP vs OLAP:

| Feature | OLTP | OLAP |
|---------|------|------|
| Purpose | Transactions | Analytics |
| Data Volume | Small (per query) | Large (bulk) |
| Speed | Milliseconds | Seconds to minutes |
| Examples | MySQL, PostgreSQL | Snowflake, BigQuery |

> 📌 **Example:** A banking system recording every deposit and withdrawal uses OLTP.

---

## 🔷 SQL

**SQL** = *Structured Query Language* — the universal language to communicate with relational databases.

### Core Concepts:

```sql
-- DDL: Define structure
CREATE TABLE users (id INT, name VARCHAR(100), age INT);

-- DML: Manipulate data
INSERT INTO users VALUES (1, 'Alice', 28);
UPDATE users SET age = 29 WHERE id = 1;
DELETE FROM users WHERE id = 1;

-- DQL: Query data
SELECT name, age FROM users WHERE age > 25 ORDER BY age DESC;

-- Joins
SELECT u.name, o.product
FROM users u
JOIN orders o ON u.id = o.user_id;

-- Aggregations
SELECT department, COUNT(*), AVG(salary)
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;
```

### Must-Know SQL Topics:
- `SELECT`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`
- `JOINs` — INNER, LEFT, RIGHT, FULL
- Window Functions — `ROW_NUMBER()`, `RANK()`, `LAG()`, `LEAD()`
- Subqueries & CTEs (`WITH` clause)
- Indexes & Query Optimization

---

## 🔷 Hadoop & HDFS

### Hadoop
**Apache Hadoop** is an open-source framework for **distributed storage and processing** of large datasets across clusters of commodity hardware.

**Core Components:**
| Component | Role |
|-----------|------|
| **HDFS** | Distributed file storage |
| **MapReduce** | Distributed computation |
| **YARN** | Resource management |
| **Common** | Shared utilities |

---

### HDFS — Hadoop Distributed File System

HDFS splits large files into **blocks (default 128MB)** and distributes them across multiple **DataNodes**, with a **NameNode** tracking the metadata.

```
          ┌─────────────┐
          │  NameNode   │  ← Metadata (file locations)
          └──────┬──────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐
│DataNode│  │DataNode│  │DataNode│
│ Block1 │  │ Block2 │  │ Block3 │
└────────┘  └────────┘  └────────┘
```

**Key Features:**
- ✅ Fault tolerant — data replicated 3x by default
- ✅ Designed for large files (GBs to PBs)
- ✅ Write-once, read-many pattern
- ❌ Not suitable for low-latency access or small files

---

## 🔷 MapReduce

**MapReduce** is a programming model for processing large datasets in **parallel across a distributed cluster**.

### How it works:

```
Input Data
    │
    ▼
┌─────────┐     Splits data into key-value pairs
│   MAP   │  →  (word, 1), (word, 1), (data, 1)
└─────────┘
    │
    ▼
┌──────────┐    Groups by key
│ SHUFFLE  │  → (word, [1,1]), (data, [1])
└──────────┘
    │
    ▼
┌──────────┐    Aggregates
│  REDUCE  │  → (word, 2), (data, 1)
└──────────┘
    │
    ▼
Output Result
```

> 💡 MapReduce is powerful but **slow** due to disk I/O between steps. This is why **Apache Spark** was created.

---

## 🔷 Apache Spark

**Apache Spark** is a fast, unified analytics engine for **large-scale data processing**, running computations **in-memory** (up to 100x faster than MapReduce).

### Key Features:
- ⚡ In-memory processing (no disk I/O between steps)
- 🔄 Supports batch, streaming, ML, and graph processing
- 🌐 Runs on Hadoop, Kubernetes, cloud, or standalone
- 🐍 APIs in Python (PySpark), Scala, Java, R, SQL

### Spark Architecture:
```
Driver Program
      │
      ▼
  Spark Context
      │
  ┌───┴───┐
  │ Cluster Manager (YARN / Kubernetes / Standalone)
  └───┬───┘
      │
  ┌───┴──────────────────┐
  │  Worker Nodes        │
  │  ┌──────┐ ┌──────┐   │
  │  │Execut│ │Execut│   │
  │  └──────┘ └──────┘   │
  └──────────────────────┘
```

### Quick PySpark Example:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Demo").getOrCreate()

df = spark.read.csv("data.csv", header=True, inferSchema=True)
df.filter(df["age"] > 25).groupBy("city").count().show()
```

### Spark Ecosystem:
| Module | Purpose |
|--------|---------|
| **Spark SQL** | Structured data & SQL queries |
| **Spark Streaming** | Real-time data streams |
| **MLlib** | Machine learning at scale |
| **GraphX** | Graph computation |

---

## 🔷 Apache Flink

**Apache Flink** is a **real-time stream processing** framework designed for **stateful computations** over unbounded (streaming) and bounded (batch) data.

### Spark vs Flink:
| Feature | Apache Spark | Apache Flink |
|---------|-------------|-------------|
| Primary Model | Batch (micro-batch streaming) | True streaming |
| Latency | Seconds (micro-batch) | Milliseconds |
| State Management | Limited | Advanced (built-in) |
| Use Case | ETL, ML, analytics | Real-time pipelines, event-driven |

### Key Concepts:
- **DataStream API** — for unbounded streaming data
- **Table API / SQL** — declarative queries on streams
- **Stateful Processing** — remembers context across events
- **Event Time** — processes events based on when they *occurred*, not arrived
- **Watermarks** — handles late-arriving data gracefully

```java
// Flink Java Example
DataStream<String> stream = env.addSource(new KafkaSource<>(...));
stream
  .keyBy(event -> event.getUserId())
  .window(TumblingEventTimeWindows.of(Time.minutes(5)))
  .aggregate(new CountAggregate())
  .print();
```

---

## 🔷 Apache Kafka & Stream Processing

### Apache Kafka
**Kafka** is a distributed **event streaming platform** used for building real-time data pipelines and streaming applications.

### Core Concepts:

```
Producers  →  [ Topic / Partitions ]  →  Consumers
               ┌──────────────────┐
               │ Partition 0      │
               │ Partition 1      │  ← Messages stored in order
               │ Partition 2      │
               └──────────────────┘
                    Broker (Kafka Server)
```

| Concept | Description |
|---------|-------------|
| **Producer** | Sends messages to a topic |
| **Consumer** | Reads messages from a topic |
| **Topic** | Named stream/category of messages |
| **Partition** | Unit of parallelism within a topic |
| **Broker** | Kafka server that stores messages |
| **Offset** | Position of a message in a partition |
| **Consumer Group** | Multiple consumers sharing partitions |

### Why Kafka?
- ✅ High throughput — millions of messages/second
- ✅ Fault tolerant — replication across brokers
- ✅ Durable — messages stored on disk (configurable retention)
- ✅ Decouples producers and consumers
- ✅ Replay — consumers can rewind and re-read messages

### Stream Processing with Kafka:
```
IoT Sensors  ─┐
Web Clicks   ─┼──►  Kafka  ──►  Flink / Spark Streaming  ──►  Dashboard
App Logs     ─┘                                             ──►  Database
                                                            ──►  Alerts
```

---

## 🗺️ Data Engineering Roadmap

```
1. Learn SQL (PostgreSQL)
        ↓
2. Learn Python (Pandas, PySpark)
        ↓
3. Understand Databases (OLTP vs OLAP)
        ↓
4. Learn Hadoop / HDFS concepts
        ↓
5. Master Apache Spark (PySpark)
        ↓
6. Learn Kafka + Stream Processing
        ↓
7. Explore Apache Flink
        ↓
8. Learn Orchestration (Airflow)
        ↓
9. Cloud Platforms (AWS / GCP / Azure)
        ↓
10. Build Real Projects 🚀
```

---

## 🛠️ Tools & Technologies Overview

| Category | Tools |
|----------|-------|
| **Language** | Python, SQL, Scala |
| **Batch Processing** | Apache Spark, Hadoop MapReduce |
| **Stream Processing** | Apache Kafka, Apache Flink, Spark Streaming |
| **Storage** | HDFS, Amazon S3, Google Cloud Storage |
| **Databases** | PostgreSQL, MySQL, MongoDB, Cassandra |
| **Data Warehouse** | Snowflake, BigQuery, Redshift |
| **Orchestration** | Apache Airflow, Prefect, Dagster |
| **Cloud** | AWS, GCP, Azure |

---

## 📖 Resources to Learn More

- 📘 [Apache Spark Official Docs](https://spark.apache.org/docs/latest/)
- 📘 [Apache Kafka Official Docs](https://kafka.apache.org/documentation/)
- 📘 [Apache Flink Official Docs](https://nightlies.apache.org/flink/flink-docs-stable/)
- 📘 [Hadoop Official Docs](https://hadoop.apache.org/docs/current/)
- 🎓 [Data Engineering Zoomcamp (Free)](https://github.com/DataTalksClub/data-engineering-zoomcamp)
- 🎓 [Mode SQL Tutorial](https://mode.com/sql-tutorial/)




