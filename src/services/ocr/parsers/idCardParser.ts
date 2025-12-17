
import { IdCardData } from '../types';

export class IdCardParser {
    parse(text: string): IdCardData {
        // TODO: Implement actual parsing logic
        // For now, return empty data structure or basic regex matches
        return {
            name: '',
            rrn: '',
            address: '',
            issueDate: ''
        };
    }
}

export const idCardParser = new IdCardParser();
