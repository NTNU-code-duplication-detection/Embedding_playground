# BigCloneBench Dataset Setup Guide (Docker + PostgreSQL)
This guide explains how to download and use the BigCloneBench v2 dataset in a PostgreSQL database running inside a Docker container.

## 1. Prerequisites
- Docker installed and running
- `psql` client (optional if you want to access PostgreSQL from host)
- Sufficient disk space for the dataset (~15 GB reccomended)

## 2. Download the Dataset
Download the PostgreSQL version of BigCloneBench v2 from the official repository

[BigCloneBench v2 ProsgreSQL dataset](https://github.com/clonebench/BigCloneBench?tab=readme-ov-file#bigclonebench-version-2-use-this-version)

Organize your datasets folder as follows:
```bash
~/datasets/
└── bcb/          # BigCloneBench dataset SQL file should be here
```

## 3. Create a Foler for Docker PostgreSQL Data
This folder will persist database data across Docker container restarts:

```bash
mkdir -p ~/docker_postgres_data
```

## 4. Run PostgreSQL in Docker
Start a PostgreSQL container using Docker:
```bash
docker run -d \
  --name bigclonebench-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -v ~/docker_postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15
```

#### Tips:
* `-v ~/docker_postgres_data:/var/lib/postgresql/data` ensures data persists even if the container is removed.
* `-p 5432:5432` maps the container's PostgreSQL port to yout local machine.
* Container runs in the background with `-d`.

## 5. Copy Dataset into Docker Container
```bash
docker cp ~/datasets/bcb bigclonebench-db:/bcb.sql
```

## 6. Enter the Docker Container
```bash
docker exec -it bigclonebench-db bash
```

## 7. Create Database
```bash
psql -U postgres
```

Create the database:

```sql
CREATE DATABSE bigclonebench;
\q
```

## 8. Load the Dataset
Load the dataset SQL file into PostgreSQL:

```bash
psql -U postgres -d bigclonebench -f /bcb.sql
```
⚠️ This may take several minutes depending on your system and disk speed.

## 9. Verify the Dataset
Enter the databse to check tables:
```bash
psql -U postgres -d bigclonebench
```
List tables:
```sql
\dt
```

## 10. Access PostgreSQL from Host (Optional)
If you want to connect from your host machine:
```bash
psql -h localhost -p 5432 -U postgres -d bigclonebench
```

## Tips & Best Practices
* Always persist data using `-v ~/docker_postgres_data:/var/lib/postgresql/data` to avoid losing it.
* If the container crashes, restart it with:
```
docker start bigclonebench-db
```
* For large datasets, use SSD storage for faster import.
* Consider increasing PostgreSQL memory settings if queries are slow.

---
This setup gives you a fully isolated PostgreSQL instance with BigCloneBench ready for querying and analysis.