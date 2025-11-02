#!/usr/bin/env python3
"""
High-Quality Synthetic Dataset Generator for SQL Injection Detection
Generates massive realistic datasets with intelligent payload insertion
"""

import argparse
import json
import random
import csv
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import urllib.parse
import base64
import math
import sys


class DatasetGenerator:
    """Generate high-quality synthetic traffic dataset"""
    
    def __init__(self, args):
        self.args = args
        self.payloads = []
        self.benign_templates = []
        self.used_hashes = {}
        self.stats = {
            "benign_count": 0,
            "malicious_count": 0,
            "duplicates_rejected": 0,
            "attack_types": {},
            "insertion_points": {},
        }
        self.inspection_samples = []
        
        # IP pool for bot simulation
        self.bot_ips = [
            f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
            for _ in range(int(args.num_malicious * args.bot_ip_ratio))
        ]
        self.normal_ips = [
            f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            for _ in range(500)
        ]
        
        # Request templates
        self.templates = self._init_templates()
        
    def _init_templates(self) -> List[Dict]:
        """Initialize request templates"""
        return [
            {
                "name": "GET single param",
                "weight": 30,
                "method": "GET",
                "url": "/search",
                "params": ["q"],
                "body": None,
                "headers": {},
            },
            {
                "name": "GET multi param",
                "weight": 20,
                "method": "GET",
                "url": "/product",
                "params": ["id", "ref"],
                "body": None,
                "headers": {},
            },
            {
                "name": "GET numeric id",
                "weight": 15,
                "method": "GET",
                "url": "/item",
                "params": ["id"],
                "body": None,
                "headers": {},
            },
            {
                "name": "POST form",
                "weight": 15,
                "method": "POST",
                "url": "/comment",
                "params": [],
                "body": "username={user}&comment={comment}",
                "headers": {"Content-Type": "application/x-www-form-urlencoded"},
            },
            {
                "name": "POST JSON",
                "weight": 10,
                "method": "POST",
                "url": "/api/search",
                "params": [],
                "body": '{{"user":"{user}","query":"{query}"}}',
                "headers": {"Content-Type": "application/json"},
            },
            {
                "name": "POST JSON nested",
                "weight": 5,
                "method": "POST",
                "url": "/api/filter",
                "params": [],
                "body": '{{"filters":{{"category":"{cat}","min_price":{min}}}}}',
                "headers": {"Content-Type": "application/json"},
            },
            {
                "name": "Path param",
                "weight": 3,
                "method": "GET",
                "url": "/item/{item_id}/details",
                "params": [],
                "body": None,
                "headers": {},
            },
            {
                "name": "Cookie injection",
                "weight": 1,
                "method": "GET",
                "url": "/profile",
                "params": [],
                "body": None,
                "headers": {"Cookie": "session={sid}; prefs={prefs}"},
            },
            {
                "name": "Header injection",
                "weight": 1,
                "method": "GET",
                "url": "/api/data",
                "params": [],
                "body": None,
                "headers": {"X-User-Data": "{header_payload}"},
            },
        ]
    
    def load_payloads(self):
        """Load payloads from JSON file"""
        payload_file = "recon/payloads.json"
        
        if not os.path.exists(payload_file):
            print(f"❌ Error: {payload_file} not found!")
            print("   Please run: python recon/generate_payloads.py")
            sys.exit(1)
        
        with open(payload_file, 'r') as f:
            self.payloads = json.load(f)
        
        print(f"✓ Loaded {len(self.payloads)} payloads from {payload_file}")
    
    def load_benign_templates(self):
        """Load or generate benign query templates"""
        benign_file = "data/benign_queries.txt"
        
        if os.path.exists(benign_file):
            with open(benign_file, 'r') as f:
                self.benign_templates = [line.strip() for line in f if line.strip()]
        
        # Generate default templates if file doesn't exist or is too small
        if len(self.benign_templates) < 100:
            self.benign_templates = self._generate_default_benign_templates()
            
            # Save for future use
            os.makedirs("data", exist_ok=True)
            with open(benign_file, 'w') as f:
                f.write('\n'.join(self.benign_templates))
        
        print(f"✓ Loaded {len(self.benign_templates)} benign templates")
    
    def _generate_default_benign_templates(self) -> List[str]:
        """Generate default benign query templates"""
        queries = []
        
        # Search terms
        search_terms = [
            "laptop", "phone", "camera", "book", "music", "movie", "game",
            "shoes", "watch", "bag", "shirt", "pants", "jacket", "hat",
            "table", "chair", "lamp", "bed", "sofa", "desk", "shelf",
            "car", "bike", "scooter", "helmet", "tire", "battery",
            "python tutorial", "javascript guide", "react documentation",
            "mysql database", "web development", "machine learning",
            "fitness tips", "healthy recipes", "workout plans",
            "travel destinations", "hotel booking", "flight tickets",
        ]
        
        for term in search_terms:
            queries.extend([
                term,
                term.upper(),
                term.capitalize(),
                f"{term} review",
                f"best {term}",
                f"{term} price",
                f"{term} 2025",
            ])
        
        # Usernames
        usernames = [
            "john_doe", "jane_smith", "admin", "user123", "alice",
            "bob", "charlie", "david", "emily", "frank",
        ]
        
        queries.extend(usernames)
        
        # Comments
        comments = [
            "Great product!",
            "Love it",
            "Not bad",
            "Could be better",
            "Excellent service",
            "Fast delivery",
            "Highly recommended",
            "Will buy again",
        ]
        
        queries.extend(comments)
        
        # Numbers and IDs
        for i in range(1, 101):
            queries.append(str(i))
        
        return queries
    
    def choose_template(self) -> Dict:
        """Choose template based on weights"""
        total_weight = sum(t["weight"] for t in self.templates)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for template in self.templates:
            cumulative += template["weight"]
            if r <= cumulative:
                return template.copy()
        
        return self.templates[0].copy()
    
    def generate_benign_request(self) -> Dict:
        """Generate a benign request"""
        template = self.choose_template()
        
        # Fill parameters with benign values
        filled_params = {}
        for param in template.get("params", []):
            filled_params[param] = random.choice(self.benign_templates)
        
        # Fill body
        body = template.get("body")
        if body:
            if "user" in body:
                body = body.replace("{user}", random.choice(self.benign_templates))
            if "comment" in body:
                body = body.replace("{comment}", random.choice(self.benign_templates))
            if "query" in body:
                body = body.replace("{query}", random.choice(self.benign_templates))
            if "cat" in body:
                body = body.replace("{cat}", random.choice(["electronics", "books", "clothing"]))
            if "min" in body:
                body = body.replace("{min}", str(random.randint(10, 1000)))
        
        # Fill URL path parameters
        url = template["url"]
        if "{item_id}" in url:
            url = url.replace("{item_id}", str(random.randint(1, 10000)))
        
        # Fill headers
        headers = template.get("headers", {}).copy()
        if "Cookie" in headers:
            headers["Cookie"] = headers["Cookie"].replace(
                "{sid}", hashlib.md5(str(random.random()).encode()).hexdigest()
            ).replace("{prefs}", "lang=en")
        
        if "X-User-Data" in headers:
            headers["X-User-Data"] = headers["X-User-Data"].replace(
                "{header_payload}", random.choice(self.benign_templates)
            )
        
        # Build raw query
        raw_query = self._build_raw_query(
            template["method"], url, filled_params, body, headers
        )
        
        return {
            "timestamp": self._generate_timestamp(),
            "source_ip": random.choice(self.normal_ips),
            "method": template["method"],
            "url": url,
            "params": json.dumps(filled_params) if filled_params else "",
            "body": body or "",
            "headers": json.dumps(headers) if headers else "",
            "raw_query": raw_query,
            "is_malicious": 0,
            "attack_type": "",
            "insertion_point": "",
            "mutation_type": "",
            "orig_payload_id": "",
            "difficulty_score": 0.0,
        }
    
    def generate_malicious_request(self) -> Dict:
        """Generate a malicious request with intelligent payload insertion"""
        template = self.choose_template()
        
        # Choose a payload
        payload_entry = random.choice(self.payloads)
        attack_type = payload_entry["attack_type"]
        
        # Choose base or variant
        if random.random() < 0.3:
            payload = payload_entry["base_payload"]
            mutation_type = "none"
        else:
            if payload_entry.get("variants"):
                payload = random.choice(payload_entry["variants"])
                mutation_type = random.choice([
                    "url_encode", "double_encode", "base64", "hex",
                    "comment_insert", "case_obf", "whitespace_obf"
                ])
            else:
                payload = payload_entry["base_payload"]
                mutation_type = "none"
        
        # Apply additional mutations
        for _ in range(random.randint(0, self.args.augment_multiplier)):
            payload = self._apply_mutation(payload)
        
        # Choose insertion point
        insertion_point = self._choose_insertion_point(template, attack_type)
        
        # Insert payload
        filled_params = {}
        body = template.get("body")
        url = template["url"]
        headers = template.get("headers", {}).copy()
        
        if insertion_point == "query_param":
            # Insert into query parameter
            target_param = random.choice(template.get("params", ["q"]))
            for param in template.get("params", []):
                if param == target_param:
                    filled_params[param] = self._adapt_payload_to_param(param, payload)
                else:
                    filled_params[param] = random.choice(self.benign_templates)
        
        elif insertion_point == "body_form":
            # Insert into form body
            if body and "{" in body:
                fields = ["user", "comment", "query"]
                target_field = random.choice([f for f in fields if f"{{{f}}}" in body])
                body = body.replace(f"{{{target_field}}}", self._escape_for_form(payload))
                
                # Fill other fields
                for field in fields:
                    if field != target_field and f"{{{field}}}" in body:
                        body = body.replace(f"{{{field}}}", random.choice(self.benign_templates))
        
        elif insertion_point == "body_json":
            # Insert into JSON body (carefully to not break JSON)
            if body and "{" in body:
                # For JSON, insert as string value
                if '"{query}"' in body:
                    safe_payload = payload.replace('"', '\\"').replace('\n', '\\n')
                    body = body.replace('"{query}"', f'"{safe_payload}"')
                elif '"{user}"' in body:
                    safe_payload = payload.replace('"', '\\"').replace('\n', '\\n')
                    body = body.replace('"{user}"', f'"{safe_payload}"')
                elif '"{cat}"' in body:
                    safe_payload = payload.replace('"', '\\"').replace('\n', '\\n')
                    body = body.replace('"{cat}"', f'"{safe_payload}"')
                
                # Fill numeric fields with safe values
                if "{min}" in body:
                    body = body.replace("{min}", str(random.randint(1, 1000)))
        
        elif insertion_point == "path":
            # Insert into path
            if "{item_id}" in url:
                url = url.replace("{item_id}", urllib.parse.quote(payload))
        
        elif insertion_point == "header":
            # Insert into header
            if "Cookie" in headers:
                headers["Cookie"] = headers["Cookie"].replace("{sid}", payload[:32])
            elif "X-User-Data" in headers:
                headers["X-User-Data"] = headers["X-User-Data"].replace(
                    "{header_payload}", payload
                )
        
        # Fill remaining placeholders
        if body:
            for placeholder in ["{user}", "{comment}", "{query}", "{cat}"]:
                if placeholder in body:
                    body = body.replace(placeholder, random.choice(self.benign_templates))
            if "{min}" in body:
                body = body.replace("{min}", str(random.randint(1, 1000)))
        
        if "Cookie" in headers and "{" in headers["Cookie"]:
            headers["Cookie"] = headers["Cookie"].replace(
                "{sid}", hashlib.md5(str(random.random()).encode()).hexdigest()
            ).replace("{prefs}", "lang=en")
        
        # Calculate difficulty score
        difficulty = self._calculate_difficulty(payload, mutation_type, insertion_point)
        
        # Build raw query
        raw_query = self._build_raw_query(
            template["method"], url, filled_params, body, headers
        )
        
        # Choose IP (bot or normal)
        if random.random() < self.args.bot_ip_ratio:
            source_ip = random.choice(self.bot_ips)
        else:
            source_ip = random.choice(self.normal_ips)
        
        return {
            "timestamp": self._generate_timestamp(),
            "source_ip": source_ip,
            "method": template["method"],
            "url": url,
            "params": json.dumps(filled_params) if filled_params else "",
            "body": body or "",
            "headers": json.dumps(headers) if headers else "",
            "raw_query": raw_query,
            "is_malicious": 1,
            "attack_type": attack_type,
            "insertion_point": insertion_point,
            "mutation_type": mutation_type,
            "orig_payload_id": payload_entry["payload_id"],
            "difficulty_score": difficulty,
        }
    
    def _choose_insertion_point(self, template: Dict, attack_type: str) -> str:
        """Choose appropriate insertion point for attack"""
        available_points = []
        
        if template.get("params"):
            available_points.append("query_param")
        
        if template.get("body"):
            if "application/json" in template.get("headers", {}).get("Content-Type", ""):
                available_points.append("body_json")
            else:
                available_points.append("body_form")
        
        if "{item_id}" in template["url"]:
            available_points.append("path")
        
        if "Cookie" in template.get("headers", {}) or "X-User-Data" in template.get("headers", {}):
            available_points.append("header")
        
        if not available_points:
            available_points = ["query_param"]
        
        # Some attack types prefer certain insertion points
        if attack_type == "header-injection" and "header" in available_points:
            return "header"
        elif attack_type == "path-injection" and "path" in available_points:
            return "path"
        
        return random.choice(available_points)
    
    def _adapt_payload_to_param(self, param: str, payload: str) -> str:
        """Adapt payload to parameter type"""
        # If parameter looks numeric, try to keep it numeric-ish
        if param in ["id", "page", "price", "min", "max", "count"]:
            # For numeric params, use numeric-context payloads
            if "OR" in payload.upper():
                # Keep OR-based but ensure it starts with a number
                if not payload[0].isdigit():
                    return f"1 {payload}"
            return payload
        
        return payload
    
    def _escape_for_form(self, payload: str) -> str:
        """Escape payload for form body"""
        return urllib.parse.quote_plus(payload)
    
    def _apply_mutation(self, payload: str) -> str:
        """Apply a random mutation"""
        mutations = [
            lambda p: p.replace(" ", "/**/"),
            lambda p: p.replace("OR", "Or") if "OR" in p else p,
            lambda p: p.replace("SELECT", "SeLeCt") if "SELECT" in p else p,
            lambda p: p.replace("UNION", "UnIoN") if "UNION" in p else p,
            lambda p: p + "%00" if len(p) < 100 else p,
            lambda p: p.replace("--", "#") if "--" in p else p,
            lambda p: p.replace(" ", "+") if random.random() < 0.5 else p,
        ]
        
        mutation = random.choice(mutations)
        try:
            return mutation(payload)
        except:
            return payload
    
    def _calculate_difficulty(self, payload: str, mutation: str, insertion: str) -> float:
        """Calculate difficulty score for detection"""
        score = 0.0
        
        # Base difficulty from mutation
        mutation_scores = {
            "none": 0.1,
            "url_encode": 0.3,
            "double_encode": 0.6,
            "base64": 0.7,
            "hex": 0.7,
            "comment_insert": 0.5,
            "case_obf": 0.4,
            "whitespace_obf": 0.4,
        }
        score += mutation_scores.get(mutation, 0.3)
        
        # Insertion point difficulty
        insertion_scores = {
            "query_param": 0.1,
            "body_form": 0.2,
            "body_json": 0.3,
            "path": 0.4,
            "header": 0.5,
        }
        score += insertion_scores.get(insertion, 0.2)
        
        # Payload complexity
        if len(payload) > 100:
            score += 0.2
        if payload.count("/*") > 2:
            score += 0.1
        if "base64" in mutation or "hex" in mutation:
            score += 0.2
        
        return min(score, 1.0)
    
    def _build_raw_query(self, method: str, url: str, params: Dict,
                         body: Optional[str], headers: Dict) -> str:
        """Build raw query string representation"""
        parts = [f"{method} {url}"]
        
        if params:
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            parts[0] += f"?{query_string}"
        
        parts[0] += " HTTP/1.1"
        
        for key, value in headers.items():
            parts.append(f"{key}: {value}")
        
        if body:
            parts.append("")
            parts.append(body)
        
        return "\n".join(parts)
    
    def _generate_timestamp(self) -> str:
        """Generate realistic timestamp"""
        start = datetime.strptime(self.args.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.args.end_date, "%Y-%m-%d")
        
        # Use sine wave for time distribution (more traffic during work hours)
        total_seconds = int((end - start).total_seconds())
        random_seconds = random.randint(0, total_seconds)
        
        # Add daily pattern
        hour_of_day = (random_seconds % 86400) / 3600
        time_weight = (math.sin((hour_of_day - 6) * math.pi / 12) + 1) / 2
        
        if random.random() > time_weight:
            random_seconds = random.randint(0, total_seconds)
        
        timestamp = start + timedelta(seconds=random_seconds)
        return timestamp.isoformat() + "Z"
    
    def _get_hash(self, row: Dict) -> str:
        """Get hash of raw query"""
        return hashlib.md5(row["raw_query"].encode()).hexdigest()
    
    def _is_duplicate(self, row: Dict) -> bool:
        """Check if row is duplicate"""
        row_hash = self._get_hash(row)
        
        if row_hash in self.used_hashes:
            self.used_hashes[row_hash] += 1
            if self.used_hashes[row_hash] > self.args.max_duplicates:
                return True
        else:
            self.used_hashes[row_hash] = 1
        
        return False
    
    def _update_stats(self, row: Dict):
        """Update generation statistics"""
        if row["is_malicious"]:
            self.stats["malicious_count"] += 1
            attack_type = row["attack_type"]
            self.stats["attack_types"][attack_type] = \
                self.stats["attack_types"].get(attack_type, 0) + 1
            
            insertion = row["insertion_point"]
            self.stats["insertion_points"][insertion] = \
                self.stats["insertion_points"].get(insertion, 0) + 1
        else:
            self.stats["benign_count"] += 1
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print(f"\n{'='*70}")
        print("DATASET GENERATION")
        print(f"{'='*70}")
        print(f"Benign records:     {self.args.num_benign:,}")
        print(f"Malicious records:  {self.args.num_malicious:,}")
        print(f"Total records:      {self.args.num_benign + self.args.num_malicious:,}")
        print(f"Chunk size:         {self.args.chunk_size:,}")
        print(f"Augment multiplier: {self.args.augment_multiplier}")
        print(f"{'='*70}\n")
        
        os.makedirs("data", exist_ok=True)
        
        csv_file = open(self.args.out_csv, 'w', newline='', encoding='utf-8')
        jl_file = open(self.args.out_jl, 'w', encoding='utf-8')
        
        fieldnames = [
            "timestamp", "source_ip", "method", "url", "params", "body",
            "headers", "raw_query", "is_malicious", "attack_type",
            "insertion_point", "mutation_type", "orig_payload_id", "difficulty_score"
        ]
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        total_records = self.args.num_benign + self.args.num_malicious
        
        # Create schedule
        schedule = (
            [0] * self.args.num_benign +
            [1] * self.args.num_malicious
        )
        random.shuffle(schedule)
        
        start_time = datetime.now()
        
        for i, is_malicious in enumerate(schedule):
            try:
                if is_malicious:
                    row = self.generate_malicious_request()
                else:
                    row = self.generate_benign_request()
                
                # Check for duplicates
                if self._is_duplicate(row):
                    self.stats["duplicates_rejected"] += 1
                    continue
                
                # Write to files
                csv_writer.writerow(row)
                jl_file.write(json.dumps(row) + "\n")
                
                # Update stats
                self._update_stats(row)
                
                # Collect samples for inspection
                if is_malicious and len(self.inspection_samples) < self.args.inspect_sample_size:
                    self.inspection_samples.append(row)
                
                # Progress
                if (i + 1) % self.args.chunk_size == 0:
                    progress = (i + 1) / total_records * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = (i + 1) / elapsed
                    eta = (total_records - i - 1) / rate
                    
                    print(f"Progress: {i+1:,}/{total_records:,} ({progress:.1f}%) | "
                          f"Rate: {rate:.0f} rec/s | ETA: {eta:.0f}s")
                    
                    # Flush files
                    csv_file.flush()
                    jl_file.flush()
            
            except Exception as e:
                print(f"✗ Error generating record {i}: {str(e)}")
                continue
        
        csv_file.close()
        jl_file.close()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print("GENERATION COMPLETED")
        print(f"{'='*70}")
        print(f"Total time:         {elapsed:.1f}s")
        print(f"Records generated:  {self.stats['benign_count'] + self.stats['malicious_count']:,}")
        print(f"Benign:             {self.stats['benign_count']:,}")
        print(f"Malicious:          {self.stats['malicious_count']:,}")
        print(f"Duplicates rejected: {self.stats['duplicates_rejected']:,}")
        print(f"{'='*70}\n")
        
        # Save reports
        self._save_reports()
        
        # Convert to Parquet if requested
        if self.args.to_parquet:
            self._convert_to_parquet()
    
    def _save_reports(self):
        """Save generation reports"""
        # Generation report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": vars(self.args),
            "stats": self.stats,
            "files": {
                "csv": self.args.out_csv,
                "jsonlines": self.args.out_jl,
            }
        }
        
        with open("data/generation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Generation report: data/generation_report.json")
        
        # Inspection samples
        with open("data/inspection_samples.json", 'w') as f:
            json.dump(self.inspection_samples, f, indent=2)
        
        print(f"✓ Inspection samples: data/inspection_samples.json ({len(self.inspection_samples)} samples)")
        
        # Update summary
        with open("deliverables/summary.txt", 'a') as f:
            f.write(f"\n[{datetime.utcnow().isoformat()}] DATASET GENERATION: SUCCESS\n")
            f.write(f"  CSV: {self.args.out_csv}\n")
            f.write(f"  JSONLines: {self.args.out_jl}\n")
            f.write(f"  Total records: {self.stats['benign_count'] + self.stats['malicious_count']:,}\n")
            f.write(f"  Benign: {self.stats['benign_count']:,}\n")
            f.write(f"  Malicious: {self.stats['malicious_count']:,}\n")
            f.write(f"  Attack types: {len(self.stats['attack_types'])}\n")
            for attack_type, count in sorted(self.stats['attack_types'].items()):
                f.write(f"    - {attack_type}: {count:,}\n")
    
    def _convert_to_parquet(self):
        """Convert CSV to Parquet"""
        try:
            import pandas as pd
            import pyarrow.parquet as pq
            
            print(f"\n🔄 Converting to Parquet...")
            
            # Read in chunks to save memory
            parquet_file = self.args.out_csv.replace('.csv', '.parquet')
            
            chunks = pd.read_csv(self.args.out_csv, chunksize=self.args.chunk_size)
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    chunk.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
                else:
                    chunk.to_parquet(parquet_file, engine='pyarrow', compression='snappy',
                                   append=True)
                
                print(f"  Chunk {i+1} converted")
            
            print(f"✓ Parquet file: {parquet_file}")
        
        except ImportError:
            print(f"⚠️  pyarrow not installed - skipping Parquet conversion")
            print(f"   Install with: pip install pyarrow pandas")
        except Exception as e:
            print(f"✗ Parquet conversion error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-quality synthetic SQL injection dataset"
    )
    
    parser.add_argument("--num-benign", type=int, default=700000,
                       help="Number of benign records (default: 700000)")
    parser.add_argument("--num-malicious", type=int, default=300000,
                       help="Number of malicious records (default: 300000)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Chunk size for writing (default: 10000)")
    parser.add_argument("--augment-multiplier", type=int, default=5,
                       help="Augmentation multiplier (default: 5)")
    parser.add_argument("--out-csv", default="data/dataset.csv",
                       help="Output CSV file (default: data/dataset.csv)")
    parser.add_argument("--out-jl", default="data/dataset.jl",
                       help="Output JSONLines file (default: data/dataset.jl)")
    parser.add_argument("--to-parquet", action="store_true",
                       help="Convert to Parquet format")
    parser.add_argument("--bot-ip-ratio", type=float, default=0.1,
                       help="Ratio of bot IPs (default: 0.1)")
    parser.add_argument("--start-date", default="2025-01-01",
                       help="Start date for timestamps (default: 2025-01-01)")
    parser.add_argument("--end-date", default="2025-10-29",
                       help="End date for timestamps (default: 2025-10-29)")
    parser.add_argument("--max-duplicates", type=int, default=3,
                       help="Max duplicate records allowed (default: 3)")
    parser.add_argument("--inspect-sample-size", type=int, default=100,
                       help="Number of samples for manual inspection (default: 100)")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args)
    
    # Load dependencies
    generator.load_payloads()
    generator.load_benign_templates()
    
    # Generate dataset
    generator.generate_dataset()
    
    print("\n✓ Dataset generation completed successfully!\n")


if __name__ == "__main__":
    main()
