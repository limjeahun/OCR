/**
 * OCR 텍스트 교정 서비스
 * 
 * 로컬 NLP 기반 하이브리드 교정 시스템
 * - 문자 혼동 규칙
 * - N-gram 확률 기반 교정
 * - 도메인 사전 통합
 */

import { isHangul, decomposeHangul, jamoSimilarity } from './hangulUtils';
import {
    CHAR_CONFUSION,
    HANGUL_ENGLISH_CONFUSION,
    WORD_CORRECTIONS,
    BIGRAM_FREQ,
    TRIGRAM_FREQ,
    FIELD_KEYWORDS,
    REGION_SET,
} from './correctionDictionary';

export interface CorrectionResult {
    original: string;
    corrected: string;
    corrections: CorrectionDetail[];
    confidence: number;
}

export interface CorrectionDetail {
    position: number;
    original: string;
    corrected: string;
    method: 'dictionary' | 'ngram' | 'confusion' | 'keyword' | 'merge' | 'prefix';
    confidence: number;
}

export class TextCorrector {
    private correctionLog: CorrectionDetail[] = [];

    /**
     * 메인 교정 함수
     */
    correct(text: string): CorrectionResult {
        this.correctionLog = [];
        let corrected = text;

        // 0단계: 줄 병합 (분리된 키워드 복원)
        corrected = this.mergeFragmentedKeywords(corrected);

        // 1단계: 가비지 프리픽스 제거 (일법인등록번호 → 법인등록번호)
        corrected = this.removeGarbagePrefix(corrected);

        // 2단계: 영어→한글 혼동 교정 (HOA → 충청남도)
        corrected = this.correctEnglishConfusion(corrected);

        // 3단계: 단어 수준 교정 (사전 기반)
        corrected = this.correctByDictionary(corrected);

        // 4단계: N-gram 기반 교정
        corrected = this.correctByNgram(corrected);

        // 5단계: 필드 키워드 교정
        corrected = this.correctFieldKeywords(corrected);

        // 신뢰도 계산
        const confidence = this.calculateConfidence(text, corrected);

        return {
            original: text,
            corrected,
            corrections: this.correctionLog,
            confidence
        };
    }

    /**
     * 0단계: 분리된 키워드 병합
     * 
     * 예: "대\n표자" → "대표자"
     */
    private mergeFragmentedKeywords(text: string): string {
        let result = text;

        // 분리된 필드 패턴 (줄바꿈이나 공백으로 분리)
        const fragmentPatterns = [
            { pattern: /대[\s\n\r]+표자/g, replacement: '대표자' },
            { pattern: /법[\s\n\r]+인명/g, replacement: '법인명' },
            { pattern: /등[\s\n\r]+록번호/g, replacement: '등록번호' },
            { pattern: /소[\s\n\r]+재지/g, replacement: '소재지' },
            { pattern: /개[\s\n\r]+업연월일/g, replacement: '개업연월일' },
            { pattern: /사[\s\n\r]+업장/g, replacement: '사업장' },
            { pattern: /본[\s\n\r]+점/g, replacement: '본점' },
        ];

        for (const { pattern, replacement } of fragmentPatterns) {
            const match = result.match(pattern);
            if (match) {
                const position = result.search(pattern);
                result = result.replace(pattern, replacement);
                this.correctionLog.push({
                    position,
                    original: match[0],
                    corrected: replacement,
                    method: 'merge',
                    confidence: 0.9
                });
            }
        }

        // 줄 끝의 단독 "대"를 다음 줄의 "표자"와 병합
        result = result.replace(/(\n|^)대\s*\n+\s*표자/gm, '$1대표자');

        return result;
    }

    /**
     * 1단계: 가비지 프리픽스 제거
     * 
     * 예: "일법인등록번호" → "법인등록번호"
     *     "01일법인등록번호" → "법인등록번호"
     */
    private removeGarbagePrefix(text: string): string {
        let result = text;

        // 날짜 접미사 후의 필드 분리
        // "2015년12월01일법인등록번호" → "2015년12월01일\n법인등록번호"
        const dateSuffixPattern = /(\d{1,2}일)([법본사개])/g;
        result = result.replace(dateSuffixPattern, '$1\n$2');

        // 가비지 프리픽스 패턴들
        const garbagePrefixPatterns = [
            // "일법인등록번호" → "법인등록번호"
            { pattern: /(?:^|\n|\s)(일)([법번]인등[록롤]번호)/g, replacement: '\n$2' },
            // "일법인등롤번호" → "법인등록번호"
            { pattern: /([법번]인등)[롤]([번빈]호)/g, replacement: '$1록$2' },
            // "등롤번호" → "등록번호"
            { pattern: /등롤번호/g, replacement: '등록번호' },
            // "번호" 앞의 가비지 제거
            { pattern: /(?:^|\s)([일인])([법번]인명)/g, replacement: ' $2' },
        ];

        for (const { pattern, replacement } of garbagePrefixPatterns) {
            const match = result.match(pattern);
            if (match) {
                const originalMatch = match[0];
                const position = result.search(pattern);
                result = result.replace(pattern, replacement);
                this.correctionLog.push({
                    position,
                    original: originalMatch,
                    corrected: replacement.replace(/^\n/, ''),
                    method: 'prefix',
                    confidence: 0.88
                });
            }
        }

        return result;
    }

    /**
     * 2단계: 영어→한글 혼동 교정
     */
    private correctEnglishConfusion(text: string): string {
        let result = text;

        for (const [english, korean] of Object.entries(HANGUL_ENGLISH_CONFUSION)) {
            if (result.includes(english)) {
                const position = result.indexOf(english);
                result = result.replace(new RegExp(english, 'g'), korean);
                this.correctionLog.push({
                    position,
                    original: english,
                    corrected: korean,
                    method: 'confusion',
                    confidence: 0.9
                });
            }
        }

        return result;
    }

    /**
     * 2단계: 단어 수준 사전 교정
     */
    private correctByDictionary(text: string): string {
        let result = text;

        // 정렬: 긴 단어 먼저 교정 (더 구체적인 패턴 우선)
        const sortedCorrections = Object.entries(WORD_CORRECTIONS)
            .sort((a, b) => b[0].length - a[0].length);

        for (const [wrong, correct] of sortedCorrections) {
            if (result.includes(wrong)) {
                const position = result.indexOf(wrong);
                result = result.replace(new RegExp(this.escapeRegex(wrong), 'g'), correct);
                this.correctionLog.push({
                    position,
                    original: wrong,
                    corrected: correct,
                    method: 'dictionary',
                    confidence: 0.95
                });
            }
        }

        return result;
    }

    /**
     * 3단계: N-gram 기반 교정
     */
    private correctByNgram(text: string): string {
        const chars = [...text];
        const result: string[] = [...chars];

        // Bi-gram 검사
        for (let i = 0; i < chars.length - 1; i++) {
            const bigram = chars[i] + chars[i + 1];

            // 낮은 빈도의 bi-gram 찾기
            if (BIGRAM_FREQ[bigram] !== undefined && BIGRAM_FREQ[bigram] < 0.1) {
                // 교정 후보 탐색
                const correction = this.findBigramCorrection(chars[i], chars[i + 1], i);
                if (correction) {
                    result[i] = correction.char1;
                    result[i + 1] = correction.char2;
                    this.correctionLog.push({
                        position: i,
                        original: bigram,
                        corrected: correction.char1 + correction.char2,
                        method: 'ngram',
                        confidence: correction.confidence
                    });
                }
            }
        }

        // Tri-gram 검사
        for (let i = 0; i < chars.length - 2; i++) {
            const trigram = result[i] + result[i + 1] + result[i + 2];

            // 낮은 빈도의 tri-gram 교정
            if (TRIGRAM_FREQ[trigram] !== undefined && TRIGRAM_FREQ[trigram] < 0.1) {
                const correction = this.findTrigramCorrection(trigram, i);
                if (correction) {
                    result[i] = correction[0];
                    result[i + 1] = correction[1];
                    result[i + 2] = correction[2];
                }
            }
        }

        return result.join('');
    }

    /**
     * Bi-gram 교정 후보 탐색
     */
    private findBigramCorrection(
        char1: string,
        char2: string,
        position: number
    ): { char1: string; char2: string; confidence: number } | null {
        let bestCorrection = null;
        let bestScore = 0;

        // 첫 번째 문자에 대한 혼동 후보
        const candidates1 = CHAR_CONFUSION[char1] || [char1];
        // 두 번째 문자에 대한 혼동 후보
        const candidates2 = CHAR_CONFUSION[char2] || [char2];

        for (const c1 of [char1, ...candidates1]) {
            for (const c2 of [char2, ...candidates2]) {
                const candidateBigram = c1 + c2;
                const freq = BIGRAM_FREQ[candidateBigram];

                if (freq !== undefined && freq > bestScore) {
                    bestScore = freq;
                    bestCorrection = { char1: c1, char2: c2, confidence: freq };
                }
            }
        }

        // 원본보다 유의미하게 좋은 교정만 반환
        const originalFreq = BIGRAM_FREQ[char1 + char2] || 0;
        if (bestScore > originalFreq + 0.3) {
            return bestCorrection;
        }

        return null;
    }

    /**
     * Tri-gram 교정 후보 탐색
     */
    private findTrigramCorrection(trigram: string, position: number): string | null {
        // 알려진 고빈도 tri-gram 중 유사한 것 찾기
        let bestMatch: string | null = null;
        let bestScore = 0;

        for (const [correctTrigram, freq] of Object.entries(TRIGRAM_FREQ)) {
            if (freq < 0.5) continue; // 낮은 빈도는 건너뛰기

            const similarity = this.trigramSimilarity(trigram, correctTrigram);
            if (similarity > 0.6 && similarity * freq > bestScore) {
                bestScore = similarity * freq;
                bestMatch = correctTrigram;
            }
        }

        return bestMatch;
    }

    /**
     * Tri-gram 유사도 계산
     */
    private trigramSimilarity(a: string, b: string): number {
        if (a.length !== 3 || b.length !== 3) return 0;

        let score = 0;
        for (let i = 0; i < 3; i++) {
            if (a[i] === b[i]) {
                score += 1;
            } else if (isHangul(a[i]) && isHangul(b[i])) {
                score += jamoSimilarity(a[i], b[i]);
            }
        }

        return score / 3;
    }

    /**
     * 4단계: 필드 키워드 교정
     */
    private correctFieldKeywords(text: string): string {
        let result = text;

        for (const keyword of FIELD_KEYWORDS) {
            // 퍼지 매칭으로 유사한 패턴 찾기
            const fuzzyPattern = this.createFuzzyPattern(keyword);
            const matches = result.match(fuzzyPattern);

            if (matches) {
                for (const match of matches) {
                    if (match !== keyword && this.levenshtein(match, keyword) <= 2) {
                        const position = result.indexOf(match);
                        result = result.replace(match, keyword);
                        this.correctionLog.push({
                            position,
                            original: match,
                            corrected: keyword,
                            method: 'keyword',
                            confidence: 0.85
                        });
                    }
                }
            }
        }

        return result;
    }

    /**
     * 퍼지 패턴 생성 (각 문자에 대해 혼동 가능한 문자 포함)
     */
    private createFuzzyPattern(keyword: string): RegExp {
        let pattern = '';

        for (const char of keyword) {
            const alternatives = CHAR_CONFUSION[char];
            if (alternatives && alternatives.length > 0) {
                pattern += `[${this.escapeRegex(char)}${alternatives.map(a => this.escapeRegex(a)).join('')}]`;
            } else {
                pattern += this.escapeRegex(char);
            }
        }

        return new RegExp(pattern, 'g');
    }

    /**
     * Levenshtein 거리 계산
     */
    private levenshtein(a: string, b: string): number {
        if (a === b) return 0;
        if (a.length === 0) return b.length;
        if (b.length === 0) return a.length;

        const matrix: number[][] = [];

        for (let i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }

        for (let j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }

        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                if (b.charAt(i - 1) === a.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                    );
                }
            }
        }

        return matrix[b.length][a.length];
    }

    /**
     * 정규식 특수문자 이스케이프
     */
    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    /**
     * 교정 신뢰도 계산
     */
    private calculateConfidence(original: string, corrected: string): number {
        if (original === corrected) return 1.0;

        const distance = this.levenshtein(original, corrected);
        const maxLen = Math.max(original.length, corrected.length);

        // 변경 비율에 기반한 신뢰도
        const changeRatio = distance / maxLen;

        // 너무 많이 변경되면 신뢰도 감소
        if (changeRatio > 0.3) return 0.5;

        // 교정 로그의 평균 신뢰도
        if (this.correctionLog.length > 0) {
            const avgConfidence = this.correctionLog.reduce((sum, c) => sum + c.confidence, 0)
                / this.correctionLog.length;
            return avgConfidence * (1 - changeRatio * 0.5);
        }

        return 0.8;
    }
}

export const textCorrector = new TextCorrector();
