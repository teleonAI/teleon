"""
Compliance Checker - Regulatory Compliance Enforcement.

Enforces compliance with various standards (GDPR, HIPAA, PCI_DSS, etc.).
"""

from typing import Dict, Any, List
from teleon.sentinel.config import ComplianceStandard
from teleon.sentinel.pii_detector import PIIDetector
from teleon.core import StructuredLogger, LogLevel


class ComplianceChecker:
    """
    Compliance rule enforcement.
    
    Checks compliance with:
    - GDPR (General Data Protection Regulation)
    - HIPAA (Health Insurance Portability and Accountability Act)
    - PCI_DSS (Payment Card Industry Data Security Standard)
    - SOC2 (System and Organization Controls 2)
    - CCPA (California Consumer Privacy Act)
    """
    
    def __init__(self, standards: List[ComplianceStandard]):
        """
        Initialize compliance checker.
        
        Args:
            standards: List of compliance standards to enforce
        """
        self.standards = standards
        self.logger = StructuredLogger("compliance_checker", LogLevel.INFO)
        self.pii_detector = PIIDetector()
    
    def check_gdpr(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check GDPR compliance.
        
        GDPR requirements:
        - Data minimization (only collect necessary data)
        - Right to be forgotten (data deletion capability)
        - Consent tracking
        - PII protection
        
        Args:
            data: Data to check
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Extract text from data
        text = self._extract_text(data)
        
        # Check for excessive PII collection
        detected_pii = self.pii_detector.detect(text)
        if detected_pii:
            pii_count = sum(len(v) for v in detected_pii.values())
            if pii_count > 10:  # Threshold for excessive collection
                violations.append({
                    'type': 'gdpr_data_minimization',
                    'message': f'Excessive PII collection detected: {pii_count} items',
                    'pii_types': list(detected_pii.keys()),
                    'severity': 'high'
                })
        
        # Check for unencrypted sensitive data indicators
        if 'password' in text.lower() or 'secret' in text.lower():
            violations.append({
                'type': 'gdpr_security',
                'message': 'Potential unencrypted sensitive data detected',
                'severity': 'high'
            })
        
        return violations
    
    def check_hipaa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check HIPAA compliance.
        
        HIPAA requirements:
        - PHI (Protected Health Information) protection
        - Encryption requirements
        - Access control
        - Audit logging
        
        Args:
            data: Data to check
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Extract text from data
        text = self._extract_text(data)
        text_lower = text.lower()
        
        # PHI indicators
        phi_indicators = [
            'medical record', 'patient id', 'diagnosis', 'treatment',
            'prescription', 'health insurance', 'medical history',
            'symptoms', 'doctor', 'physician', 'hospital'
        ]
        
        phi_detected = any(indicator in text_lower for indicator in phi_indicators)
        
        if phi_detected:
            # Check for encryption indicators
            has_encryption = 'encrypt' in text_lower or 'encrypted' in text_lower
            
            if not has_encryption:
                violations.append({
                    'type': 'hipaa_encryption',
                    'message': 'PHI detected without encryption indicators',
                    'severity': 'critical'
                })
            
            # Check for access control
            has_access_control = 'access control' in text_lower or 'permission' in text_lower
            
            if not has_access_control:
                violations.append({
                    'type': 'hipaa_access_control',
                    'message': 'PHI detected without access control indicators',
                    'severity': 'high'
                })
        
        return violations
    
    def check_pci_dss(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check PCI_DSS compliance.
        
        PCI_DSS requirements:
        - Credit card data protection
        - Security requirements
        - Encryption of cardholder data
        
        Args:
            data: Data to check
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Extract text from data
        text = self._extract_text(data)
        
        # Check for credit card data
        detected_pii = self.pii_detector.detect(text)
        if 'credit_cards' in detected_pii:
            # Check for encryption indicators
            text_lower = text.lower()
            has_encryption = 'encrypt' in text_lower or 'tokenize' in text_lower
            
            if not has_encryption:
                violations.append({
                    'type': 'pci_dss_encryption',
                    'message': 'Credit card data detected without encryption indicators',
                    'severity': 'critical',
                    'card_count': len(detected_pii['credit_cards'])
                })
        
        return violations
    
    def check_soc2(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check SOC2 compliance.
        
        SOC2 requirements:
        - Security controls
        - Availability
        - Processing integrity
        - Confidentiality
        - Privacy
        
        Args:
            data: Data to check
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Basic SOC2 checks
        text = self._extract_text(data)
        text_lower = text.lower()
        
        # Check for security indicators
        if 'password' in text_lower and 'plaintext' in text_lower:
            violations.append({
                'type': 'soc2_security',
                'message': 'Potential plaintext password storage detected',
                'severity': 'high'
            })
        
        return violations
    
    def check_ccpa(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check CCPA compliance.
        
        CCPA requirements:
        - Consumer privacy rights
        - Data disclosure requirements
        - Opt-out mechanisms
        
        Args:
            data: Data to check
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Basic CCPA checks
        text = self._extract_text(data)
        detected_pii = self.pii_detector.detect(text)
        
        if detected_pii:
            # Check for privacy policy indicators
            text_lower = text.lower()
            has_privacy_policy = 'privacy policy' in text_lower or 'opt-out' in text_lower
            
            if not has_privacy_policy:
                violations.append({
                    'type': 'ccpa_privacy',
                    'message': 'PII detected without privacy policy indicators',
                    'severity': 'medium'
                })
        
        return violations
    
    def check_all(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all enabled compliance standards.
        
        Args:
            data: Data to check
        
        Returns:
            List of all violations
        """
        all_violations = []
        
        for standard in self.standards:
            if standard == ComplianceStandard.GDPR:
                all_violations.extend(self.check_gdpr(data))
            elif standard == ComplianceStandard.HIPAA:
                all_violations.extend(self.check_hipaa(data))
            elif standard == ComplianceStandard.PCI_DSS:
                all_violations.extend(self.check_pci_dss(data))
            elif standard == ComplianceStandard.SOC2:
                all_violations.extend(self.check_soc2(data))
            elif standard == ComplianceStandard.CCPA:
                all_violations.extend(self.check_ccpa(data))
        
        if all_violations:
            self.logger.debug(
                "Compliance violations detected",
                standards=[s.value for s in self.standards],
                violation_count=len(all_violations)
            )
        
        return all_violations
    
    def _extract_text(self, data: Any) -> str:
        """Extract text from various data types."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            texts = []
            for value in data.values():
                texts.append(self._extract_text(value))
            return ' '.join(texts)
        elif isinstance(data, list):
            texts = [self._extract_text(item) for item in data]
            return ' '.join(texts)
        else:
            return str(data)

