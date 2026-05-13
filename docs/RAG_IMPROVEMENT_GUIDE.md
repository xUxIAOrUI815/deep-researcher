# RAG绯荤粺鏀硅繘浠诲姟鎸囧鏂囨。

> 鐗堟湰: 1.0  
> 鍒涘缓鏃ユ湡: 2026-04-05  
> 鐩爣: 灏嗗綋鍓嶇殑鍩虹鐭ヨ瘑瀛樺偍鍗囩骇涓哄畬鏁寸殑RAG妫€绱㈠寮虹敓鎴愮郴缁?
---

## 鐩綍

1. [鐜扮姸鍒嗘瀽](#1-鐜扮姸鍒嗘瀽)
2. [鏀硅繘浠诲姟娓呭崟](#2-鏀硅繘浠诲姟娓呭崟)
3. [P0绾т换鍔¤瑙(#3-p0绾т换鍔¤瑙?
4. [P1绾т换鍔¤瑙(#4-p1绾т换鍔¤瑙?
5. [P2绾т换鍔¤瑙(#5-p2绾т换鍔¤瑙?
6. [瀹炴柦璺嚎鍥綸(#6-瀹炴柦璺嚎鍥?
7. [椋庨櫓璇勪及涓庡簲瀵筣(#7-椋庨櫓璇勪及涓庡簲瀵?

---

## 1. 鐜扮姸鍒嗘瀽

### 1.1 褰撳墠鏋舵瀯

```
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?                     褰撳墠鏁版嵁娴?                                  鈹?鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?                                                                鈹?鈹? MCPGateway 鈹€鈹€鈻?SmartScraper 鈹€鈹€鈻?DistillerAgent 鈹€鈹€鈻?KnowledgeManager
鈹? (鎼滅储)         (鎶撳彇+闄嶅櫔)       (鎻愬彇鍘熷瓙浜嬪疄)      (JSON鏈湴瀛樺偍)
鈹?                                                                鈹?鈹?      鈫?             鈫?               鈫?                 鈫?      鈹?鈹? SearchResult   ScrapedData      AtomicFact         StoredFact  鈹?鈹?                                                                鈹?鈹?                             鈫?                                 鈹?鈹?                       Writer (鐩存帴浣跨敤鍏ㄩ儴浜嬪疄)                   鈹?鈹?                                                                鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

### 1.2 鏍稿績闂鎬荤粨

| 闂缂栧彿 | 闂鎻忚堪 | 褰卞搷绋嬪害 | 褰撳墠鐘舵€?|
|---------|---------|---------|---------|
| P0-1 | 鍚戦噺鏁版嵁搴撴湭闆嗘垚 | 涓ラ噸 | QdrantVectorStore宸插疄鐜颁絾鏈娇鐢?|
| P0-2 | 缂轰箯鐪熸鐨凴AG妫€绱?| 涓ラ噸 | Writer鐩存帴浣跨敤鍏ㄩ儴浜嬪疄 |
| P1-1 | 缂轰箯鏂囨。鍒嗗潡绛栫暐 | 涓瓑 | 鐩存帴鎴柇鑷?000瀛楃 |
| P1-2 | 缂轰箯閲嶆帓搴忔満鍒?| 涓瓑 | 浠呮寜鍚戦噺鐩镐技搴︽帓搴?|
| P1-3 | 缂轰箯娣峰悎妫€绱?| 涓瓑 | 浠呬娇鐢ㄥ悜閲忔绱?|
| P2-1 | 缂轰箯鏉ユ簮鍙俊搴﹁瘎浼?| 杈冧綆 | confidence浠呯敱LLM涓昏鍒ゆ柇 |
| P2-2 | 缂轰箯澧為噺鏇存柊涓庣増鏈鐞?| 杈冧綆 | 浜嬪疄鏃犵増鏈巻鍙?|

### 1.3 鐜版湁璧勬簮璇勪及

| 缁勪欢 | 鏂囦欢璺緞 | 鍙鐢ㄧ▼搴?| 澶囨敞 |
|-----|---------|-----------|-----|
| QdrantVectorStore | `core/vector_store_qdrant.py` | 楂?| 瀹屾暣瀹炵幇锛屼粎闇€闆嗘垚 |
| KnowledgeManager | `core/knowledge.py` | 涓?| 闇€瑕侀噸鏋勪互鏀寔鍚戦噺鏁版嵁搴?|
| DistillerAgent | `agents/distiller.py` | 涓?| 闇€瑕佹坊鍔犲垎鍧楁敮鎸?|
| SmartScraper | `providers/scraper.py` | 楂?| 宸叉湁闄嶅櫔鍔熻兘锛屽彲鐩存帴浣跨敤 |
| EmbeddingModel | `core/knowledge.py` | 楂?| SiliconFlow API宸查泦鎴?|

---

## 2. 鏀硅繘浠诲姟娓呭崟

### 浼樺厛绾у畾涔?
- **P0 (Critical)**: 蹇呴』瀹屾垚锛岀洿鎺ュ奖鍝嶆牳蹇冨姛鑳?- **P1 (High)**: 寮虹儓寤鸿瀹屾垚锛屾樉钁楁彁鍗囨晥鏋?- **P2 (Medium)**: 寤鸿瀹屾垚锛屾彁鍗囩郴缁熷仴澹€?
### 浠诲姟鎬昏

```
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?                        浠诲姟渚濊禆鍏崇郴                                 鈹?鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?                                                                   鈹?鈹? P0-1: 鍚戦噺鏁版嵁搴撻泦鎴?                                              鈹?鈹?   鈹?                                                              鈹?鈹?   鈹溾攢鈹€鈻?P0-2: RAG妫€绱㈠疄鐜?                                         鈹?鈹?   鈹?        鈹?                                                    鈹?鈹?   鈹?        鈹斺攢鈹€鈻?P1-2: 閲嶆帓搴忔満鍒?                                鈹?鈹?   鈹?                                                              鈹?鈹? P1-1: 鏂囨。鍒嗗潡绛栫暐                                                 鈹?鈹?   鈹?                                                              鈹?鈹?   鈹斺攢鈹€鈻?P1-3: 娣峰悎妫€绱?(渚濊禆BM25绱㈠紩)                               鈹?鈹?                                                                   鈹?鈹? P2-1: 鏉ユ簮鍙俊搴﹁瘎浼?                                              鈹?鈹?                                                                   鈹?鈹? P2-2: 鐗堟湰绠＄悊                                                     鈹?鈹?                                                                   鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

---

## 3. P0绾т换鍔¤瑙?
### 3.1 P0-1: 鍚戦噺鏁版嵁搴撻泦鎴?
#### 3.1.1 闂鎻忚堪

褰撳墠 `KnowledgeManager` 浣跨敤鏈湴JSON鏂囦欢瀛樺偍浜嬪疄锛屽苟鍦ㄥ唴瀛樹腑閬嶅巻鎵€鏈夊悜閲忚绠楃浉浼煎害銆傝繖绉嶆柟寮忓瓨鍦ㄤ互涓嬮棶棰橈細

1. **鎵╁睍鎬у樊**: 鏁版嵁閲忓澶у悗鍐呭瓨鍗犵敤杩囬珮
2. **妫€绱㈡晥鐜囦綆**: O(n)澶嶆潅搴︼紝姣忔妫€绱㈤渶閬嶅巻鎵€鏈夊悜閲?3. **鎸佷箙鍖栭闄?*: JSON鏂囦欢鎹熷潖浼氬鑷存暟鎹涪澶?4. **鏃犳硶鏀寔楂樼骇鏌ヨ**: 缂轰箯杩囨护銆佽仛鍚堢瓑鍔熻兘

#### 3.1.2 鍙鎬у垎鏋?
| 缁村害 | 璇勪及 | 璇存槑 |
|-----|-----|-----|
| 鎶€鏈彲琛屾€?| 鉁?楂?| QdrantVectorStore宸插畬鏁村疄鐜?|
| 璧勬簮闇€姹?| 鉁?浣?| Qdrant鏀寔Docker閮ㄧ讲锛岃祫婧愬崰鐢ㄥ皬 |
| 鍏煎鎬?| 鉁?楂?| 鍙€氳繃閰嶇疆寮€鍏冲垏鎹㈠瓨鍌ㄥ悗绔?|
| 椋庨櫓 | 鈿狅笍 涓?| 闇€瑕佹暟鎹縼绉荤瓥鐣?|

#### 3.1.3 鏈€浼樺疄鐜版柟妗?
**鏂规閫夋嫨**: 閫傞厤鍣ㄦā寮?+ 閰嶇疆寮€鍏?
```python
# core/knowledge.py 閲嶆瀯鏂规

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.vector_store_qdrant import QdrantVectorStore
from core.config import QdrantConfig

class VectorStoreAdapter(ABC):
    """鍚戦噺瀛樺偍閫傞厤鍣ㄦ帴鍙?""
    
    @abstractmethod
    async def upsert(self, id: str, vector: List[float], payload: Dict) -> bool:
        pass
    
    @abstractmethod
    async def search(self, vector: List[float], limit: int, threshold: float) -> List[Dict]:
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        pass
    
    @abstractmethod
    async def get(self, id: str) -> Optional[Dict]:
        pass


class InMemoryVectorStore(VectorStoreAdapter):
    """鍐呭瓨鍚戦噺瀛樺偍 (褰撳墠瀹炵幇鐨勫皝瑁?"""
    
    def __init__(self):
        self._vectors: Dict[str, tuple] = {}  # id -> (vector, payload)
    
    async def upsert(self, id: str, vector: List[float], payload: Dict) -> bool:
        self._vectors[id] = (vector, payload)
        return True
    
    async def search(self, vector: List[float], limit: int, threshold: float) -> List[Dict]:
        results = []
        query_vec = np.array(vector)
        
        for id, (vec, payload) in self._vectors.items():
            score = self._cosine_similarity(query_vec, np.array(vec))
            if score >= threshold:
                results.append({"id": id, "score": score, "payload": payload})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class QdrantAdapter(VectorStoreAdapter):
    """Qdrant鍚戦噺瀛樺偍閫傞厤鍣?""
    
    def __init__(self, collection_name: str = "knowledge_facts"):
        self.client = QdrantVectorStore(
            host=QdrantConfig.HOST,
            port=QdrantConfig.PORT
        )
        self.collection_name = collection_name
    
    async def upsert(self, id: str, vector: List[float], payload: Dict) -> bool:
        return self.client.upsert_point(
            collection_name=self.collection_name,
            point_id=id,
            vector=vector,
            payload=payload
        )
    
    async def search(self, vector: List[float], limit: int, threshold: float) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=threshold
        )
        return [{"id": r[0], "score": r[1], "payload": r[2]} for r in results]
    
    async def delete(self, id: str) -> bool:
        # Qdrant闇€瑕佸疄鐜癲elete鏂规硶
        pass
    
    async def get(self, id: str) -> Optional[Dict]:
        return self.client.retrieve(self.collection_name, id)


class KnowledgeManager:
    """閲嶆瀯鍚庣殑鐭ヨ瘑绠＄悊鍣?""
    
    def __init__(
        self,
        storage_path: str = "./knowledge_data",
        use_qdrant: Optional[bool] = None
    ):
        self.storage_path = storage_path
        self.embedding_model = EmbeddingModel()
        
        # 鏍规嵁閰嶇疆閫夋嫨瀛樺偍鍚庣
        if use_qdrant is None:
            use_qdrant = QdrantConfig.USE_QDRANT
        
        if use_qdrant:
            self.vector_store = QdrantAdapter()
            self._backend_type = "qdrant"
        else:
            self.vector_store = InMemoryVectorStore()
            self._backend_type = "memory"
        
        self._stats = KnowledgeStats()
        self._conflicts: List[FactConflict] = []
        
        if self._backend_type == "memory":
            os.makedirs(storage_path, exist_ok=True)
            self._load()
```

#### 3.1.4 瀹炴柦姝ラ

| 姝ラ | 浠诲姟 | 棰勪及鏃堕棿 | 浜у嚭鐗?|
|-----|-----|---------|--------|
| 1 | 鍒涘缓VectorStoreAdapter鎶借薄绫?| 30min | `core/vector_store_adapter.py` |
| 2 | 瀹炵幇InMemoryVectorStore | 1h | 灏佽鐜版湁閫昏緫 |
| 3 | 瀹炵幇QdrantAdapter | 1h | 闆嗘垚QdrantVectorStore |
| 4 | 閲嶆瀯KnowledgeManager | 2h | 鏀寔鍚庣鍒囨崲 |
| 5 | 缂栧啓鍗曞厓娴嬭瘯 | 1h | `tests/test_vector_adapter.py` |
| 6 | 鏁版嵁杩佺Щ鑴氭湰 | 1h | `scripts/migrate_to_qdrant.py` |

#### 3.1.5 楠屾敹鏍囧噯

- [ ] 閰嶇疆 `USE_QDRANT=true` 鏃朵娇鐢≦drant瀛樺偍
- [ ] 閰嶇疆 `USE_QDRANT=false` 鏃朵娇鐢ㄥ唴瀛樺瓨鍌?- [ ] 鐜版湁娴嬭瘯鍏ㄩ儴閫氳繃
- [ ] 鏂板閫傞厤鍣ㄦ祴璇曡鐩栫巼 > 80%
- [ ] 鏀寔鏁版嵁杩佺Щ鍛戒护

---

### 3.2 P0-2: RAG妫€绱㈠疄鐜?
#### 3.2.1 闂鎻忚堪

褰撳墠 `writer_async` 鍑芥暟鐩存帴浣跨敤 `state["atomic_facts"]` 涓殑鍏ㄩ儴浜嬪疄鐢熸垚鎶ュ憡锛屾病鏈夋牴鎹姤鍛婁富棰樿繘琛岃涔夋绱€傝繖瀵艰嚧锛?
1. **涓婁笅鏂囩獥鍙ｆ氮璐?*: 涓嶇浉鍏崇殑浜嬪疄鍗犵敤token
2. **鎶ュ憡璐ㄩ噺涓嬮檷**: 鍣煶浜嬪疄骞叉壈鐢熸垚
3. **鏃犳硶澶勭悊澶ц妯℃暟鎹?*: 浜嬪疄鏁伴噺瓒呰繃涓婁笅鏂囬檺鍒?
#### 3.2.2 鍙鎬у垎鏋?
| 缁村害 | 璇勪及 | 璇存槑 |
|-----|-----|-----|
| 鎶€鏈彲琛屾€?| 鉁?楂?| 妫€绱㈤€昏緫宸插瓨鍦ㄤ簬KnowledgeManager |
| 璧勬簮闇€姹?| 鉁?浣?| 澶嶇敤鐜版湁embedding鍜屾绱?|
| 鍏煎鎬?| 鉁?楂?| 瀵圭幇鏈夋祦绋嬫棤鐮村潖鎬у彉鏇?|
| 椋庨櫓 | 鉁?浣?| 鍙€愭杩佺Щ |

#### 3.2.3 鏈€浼樺疄鐜版柟妗?
**鏂规閫夋嫨**: 鏌ヨ鎵╁睍 + 澶氶樁娈垫绱?
```python
# core/rag_retriever.py (鏂版枃浠?

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.session_knowledge import KnowledgeManager, FactStatus

@dataclass
class RetrievedContext:
    """妫€绱㈢粨鏋滀笂涓嬫枃"""
    facts: List[Dict[str, Any]]
    total_count: int
    query: str
    retrieval_strategy: str


class RAGRetriever:
    """RAG妫€绱㈠櫒"""
    
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.km = knowledge_manager
    
    async def retrieve_for_report(
        self,
        root_query: str,
        task_tree: Dict[str, Any],
        max_facts: int = 30,
        min_relevance: float = 0.3
    ) -> RetrievedContext:
        """
        涓烘姤鍛婄敓鎴愭绱㈢浉鍏充簨瀹?        
        绛栫暐:
        1. 浣跨敤鏍规煡璇㈣繘琛屽垵濮嬫绱?        2. 浣跨敤瀛愪换鍔℃煡璇㈣繘琛岃ˉ鍏呮绱?        3. 鍚堝苟鍘婚噸鍚庢寜鐩稿叧鎬ф帓搴?        """
        all_results = []
        seen_ids = set()
        
        # 闃舵1: 鏍规煡璇㈡绱?        root_results = await self.km.search_facts(
            query=root_query,
            limit=max_facts // 2,
            status_filter=FactStatus.VERIFIED
        )
        for r in root_results:
            if r["id"] not in seen_ids and r["score"] >= min_relevance:
                all_results.append(r)
                seen_ids.add(r["id"])
        
        # 闃舵2: 瀛愪换鍔℃煡璇㈣ˉ鍏呮绱?        sub_queries = self._extract_sub_queries(task_tree)
        remaining_quota = max_facts - len(all_results)
        
        for query in sub_queries[:3]:  # 鏈€澶氫娇鐢?涓瓙鏌ヨ
            if remaining_quota <= 0:
                break
            
            sub_results = await self.km.search_facts(
                query=query,
                limit=remaining_quota // len(sub_queries[:3]),
                status_filter=FactStatus.ACTIVE
            )
            
            for r in sub_results:
                if r["id"] not in seen_ids and r["score"] >= min_relevance:
                    all_results.append(r)
                    seen_ids.add(r["id"])
                    remaining_quota -= 1
        
        # 闃舵3: 鎸夌浉鍏虫€ф帓搴?        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return RetrievedContext(
            facts=all_results[:max_facts],
            total_count=len(all_results),
            query=root_query,
            retrieval_strategy="multi_stage"
        )
    
    def _extract_sub_queries(self, task_tree: Dict[str, Any]) -> List[str]:
        """浠庝换鍔℃爲鎻愬彇瀛愭煡璇?""
        queries = []
        for task_id, task in task_tree.items():
            query = task.get("query", "")
            if query and task.get("status") == "completed":
                queries.append(query)
        return queries


# core/graph.py 淇敼

async def writer_async(state: GraphState) -> GraphState:
    """浣跨敤RAG妫€绱㈢殑Writer"""
    
    from core.session_knowledge import KnowledgeManager
    from core.rag_retriever import RAGRetriever
    
    # 鍒濆鍖栨绱㈠櫒
    km = KnowledgeManager()
    retriever = RAGRetriever(km)
    
    # 鑾峰彇鏍规煡璇?    root_query = ""
    if state.get("root_task_id"):
        root_task = state["task_tree"].get(state["root_task_id"], {})
        root_query = root_task.get("query", "")
    
    # RAG妫€绱?    context = await retriever.retrieve_for_report(
        root_query=root_query,
        task_tree=state["task_tree"],
        max_facts=30
    )
    
    # 鏍煎紡鍖栦簨瀹?    facts_text = []
    for i, fact in enumerate(context.facts, 1):
        facts_text.append(
            f"[{fact['id'][:8]}] {fact['text']} "
            f"(鏉ユ簮: {fact['source_url']}, 鐩稿叧搴? {fact['score']:.2f})"
        )
    
    facts_context = "\n\n".join(facts_text)
    
    # ... 鍚庣画鎶ュ憡鐢熸垚閫昏緫
```

#### 3.2.4 瀹炴柦姝ラ

| 姝ラ | 浠诲姟 | 棰勪及鏃堕棿 | 浜у嚭鐗?|
|-----|-----|---------|--------|
| 1 | 鍒涘缓RAGRetriever绫?| 1h | `core/rag_retriever.py` |
| 2 | 瀹炵幇澶氶樁娈垫绱㈢瓥鐣?| 1.5h | retrieve_for_report鏂规硶 |
| 3 | 淇敼writer_async | 1h | 闆嗘垚RAG妫€绱?|
| 4 | 娣诲姞妫€绱㈡棩蹇?| 30min | 渚夸簬璋冭瘯鍜岃瘎浼?|
| 5 | 缂栧啓闆嗘垚娴嬭瘯 | 1h | `tests/test_rag_retriever.py` |

#### 3.2.5 楠屾敹鏍囧噯

- [ ] Writer浣跨敤妫€绱㈠埌鐨勪簨瀹炶€岄潪鍏ㄩ儴浜嬪疄
- [ ] 妫€绱㈢粨鏋滄寜鐩稿叧鎬ф帓搴?- [ ] 鏀寔澶氶樁娈垫绱㈢瓥鐣?- [ ] 妫€绱㈣繃绋嬫湁璇︾粏鏃ュ織
- [ ] 娴嬭瘯瑕嗙洊鐜?> 80%

---

## 4. P1绾т换鍔¤瑙?
### 4.1 P1-1: 鏂囨。鍒嗗潡绛栫暐

#### 4.1.1 闂鎻忚堪

褰撳墠 `DistillerAgent` 鐩存帴灏嗘枃妗ｆ埅鏂嚦8000瀛楃锛屽鑷达細

1. **淇℃伅涓㈠け**: 瓒呭嚭闀垮害鐨勫唴瀹硅涓㈠純
2. **璇箟鏂**: 鍙兘鍦ㄥ彞瀛愪腑闂存埅鏂?3. **鎻愬彇璐ㄩ噺涓嬮檷**: LLM鏃犳硶鐞嗚В涓嶅畬鏁寸殑涓婁笅鏂?
#### 4.1.2 鍙鎬у垎鏋?
| 缁村害 | 璇勪及 | 璇存槑 |
|-----|-----|-----|
| 鎶€鏈彲琛屾€?| 鉁?楂?| 鍒嗗潡绠楁硶鎴愮啛 |
| 璧勬簮闇€姹?| 鈿狅笍 涓?| 澧炲姞API璋冪敤娆℃暟 |
| 鍏煎鎬?| 鉁?楂?| 瀵逛笅娓搁€忔槑 |
| 椋庨櫓 | 鈿狅笍 涓?| 闇€瑕佸鐞嗚法鍧椾簨瀹?|

#### 4.1.3 鏈€浼樺疄鐜版柟妗?
**鏂规閫夋嫨**: 璇箟鍒嗗潡 + 婊戝姩绐楀彛

```python
# core/chunker.py (鏂版枃浠?

import re
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    """鏂囨。鍧?""
    text: str
    start_index: int
    end_index: int
    token_count: int
    overlap_with_previous: int
    source_section: Optional[str] = None


class SemanticChunker:
    """璇箟鍒嗗潡鍣?""
    
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 100
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
    
    def chunk(self, markdown: str) -> List[Chunk]:
        """
        璇箟鍒嗗潡绛栫暐:
        1. 鎸夋爣棰樺垎鍓叉枃妗?        2. 瀵规瘡涓珷鑺傝繘琛屾钀藉垎鍓?        3. 鍚堝苟灏忔钀斤紝鎷嗗垎澶ф钀?        4. 娣诲姞閲嶅彔绐楀彛
        """
        # 鎸夋爣棰樺垎鍓?        sections = self._split_by_headers(markdown)
        
        chunks = []
        current_position = 0
        
        for section_title, section_text in sections:
            section_chunks = self._chunk_section(
                section_text,
                section_title,
                current_position
            )
            chunks.extend(section_chunks)
            current_position += len(section_text)
        
        # 娣诲姞閲嶅彔
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _split_by_headers(self, markdown: str) -> List[tuple]:
        """鎸夋爣棰樺垎鍓?""
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = markdown.split('\n')
        
        sections = []
        current_title = "Introduction"
        current_text = []
        
        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                if current_text:
                    sections.append((current_title, '\n'.join(current_text)))
                current_title = match.group(2)
                current_text = []
            else:
                current_text.append(line)
        
        if current_text:
            sections.append((current_title, '\n'.join(current_text)))
        
        return sections
    
    def _chunk_section(
        self,
        text: str,
        section_title: str,
        start_pos: int
    ) -> List[Chunk]:
        """瀵圭珷鑺傝繘琛屽垎鍧?""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_start = start_pos
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            if current_tokens + para_tokens > self.max_tokens:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_index=chunk_start,
                        end_index=chunk_start + len(chunk_text),
                        token_count=current_tokens,
                        overlap_with_previous=0,
                        source_section=section_title
                    ))
                current_chunk = [para]
                current_tokens = para_tokens
                chunk_start = start_pos + text.find(para)
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        if current_chunk and current_tokens >= self.min_chunk_tokens:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_index=chunk_start,
                end_index=chunk_start + len(chunk_text),
                token_count=current_tokens,
                overlap_with_previous=0,
                source_section=section_title
            ))
        
        return chunks
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """娣诲姞閲嶅彔绐楀彛"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # 浠庡墠涓€涓潡鏈熬鎻愬彇閲嶅彔鏂囨湰
            overlap_text = self._get_overlap_text(
                prev_chunk.text,
                self.overlap_tokens
            )
            
            # 鍒涘缓甯﹂噸鍙犵殑鏂板潡
            new_text = overlap_text + '\n\n' + curr_chunk.text
            overlapped.append(Chunk(
                text=new_text,
                start_index=curr_chunk.start_index,
                end_index=curr_chunk.end_index,
                token_count=self._estimate_tokens(new_text),
                overlap_with_previous=len(overlap_text),
                source_section=curr_chunk.source_section
            ))
        
        return overlapped
    
    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """鑾峰彇閲嶅彔鏂囨湰"""
        sentences = re.split(r'(?<=[銆傦紒锛?!?])\s*', text)
        
        overlap_sentences = []
        token_count = 0
        
        for sentence in reversed(sentences):
            sent_tokens = self._estimate_tokens(sentence)
            if token_count + sent_tokens > max_tokens:
                break
            overlap_sentences.insert(0, sentence)
            token_count += sent_tokens
        
        return ' '.join(overlap_sentences)
    
    def _estimate_tokens(self, text: str) -> int:
        """浼扮畻token鏁伴噺 (涓枃绾?.5瀛楃/token锛岃嫳鏂囩害4瀛楃/token)"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


# agents/distiller.py 淇敼

class DistillerAgent:
    
    def __init__(self, ...):
        # ... 鐜版湁鍒濆鍖?        self.chunker = SemanticChunker(
            max_tokens=512,
            overlap_tokens=50
        )
    
    async def distill(self, markdown_text: str, source_url: str, task_id: Optional[str] = None) -> DistillationResult:
        """浣跨敤鍒嗗潡绛栫暐鎻愬彇浜嬪疄"""
        
        # 鍒嗗潡
        chunks = self.chunker.chunk(markdown_text)
        print(f"[DistillerAgent] Split into {len(chunks)} chunks")
        
        all_facts = []
        seen_fact_texts = set()
        
        for i, chunk in enumerate(chunks):
            # 瀵规瘡涓潡鎻愬彇浜嬪疄
            prompt = self._build_prompt(chunk.text)
            response = await self._call_api(prompt)
            chunk_facts = self._parse_facts_from_response(response, source_url, task_id)
            
            # 鍘婚噸
            for fact in chunk_facts:
                normalized = self._normalize_fact_text(fact.text)
                if normalized not in seen_fact_texts:
                    all_facts.append(fact)
                    seen_fact_texts.add(normalized)
            
            print(f"[DistillerAgent] Chunk {i+1}/{len(chunks)}: {len(chunk_facts)} facts")
        
        # 鍚堝苟鐩镐技浜嬪疄
        merged_facts = self._merge_similar_facts(all_facts)
        
        return DistillationResult(
            facts=merged_facts,
            summary=self._extract_summary(markdown_text),
            raw_response=""
        )
    
    def _normalize_fact_text(self, text: str) -> str:
        """鏍囧噯鍖栦簨瀹炴枃鏈敤浜庡幓閲?""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def _merge_similar_facts(self, facts: List[AtomicFact]) -> List[AtomicFact]:
        """鍚堝苟鐩镐技浜嬪疄"""
        # 绠€鍗曞疄鐜? 鎸夋枃鏈暱搴︽帓搴忥紝淇濈暀鏈€璇︾粏鐨勭増鏈?        unique_facts = {}
        for fact in facts:
            key = fact.text[:50]  # 浣跨敤鍓?0瀛楃浣滀负key
            if key not in unique_facts or len(fact.text) > len(unique_facts[key].text):
                unique_facts[key] = fact
        return list(unique_facts.values())
```

#### 4.1.4 瀹炴柦姝ラ

| 姝ラ | 浠诲姟 | 棰勪及鏃堕棿 | 浜у嚭鐗?|
|-----|-----|---------|--------|
| 1 | 鍒涘缓SemanticChunker绫?| 2h | `core/chunker.py` |
| 2 | 瀹炵幇鏍囬鍒嗗壊 | 1h | _split_by_headers鏂规硶 |
| 3 | 瀹炵幇璇箟鍒嗗潡 | 1.5h | _chunk_section鏂规硶 |
| 4 | 瀹炵幇閲嶅彔绐楀彛 | 1h | _add_overlap鏂规硶 |
| 5 | 淇敼DistillerAgent | 1.5h | 闆嗘垚鍒嗗潡绛栫暐 |
| 6 | 缂栧啓娴嬭瘯 | 1h | `tests/test_chunker.py` |

#### 4.1.5 楠屾敹鏍囧噯

- [ ] 鏀寔鎸夋爣棰樺垎鍓叉枃妗?- [ ] 鏀寔婊戝姩绐楀彛閲嶅彔
- [ ] 涓嶅湪鍙ュ瓙涓棿鎴柇
- [ ] 璺ㄥ潡浜嬪疄鑳芥纭悎骞?- [ ] 娴嬭瘯瑕嗙洊鐜?> 80%

---

### 4.2 P1-2: 閲嶆帓搴忔満鍒?
#### 4.2.1 闂鎻忚堪

褰撳墠妫€绱㈢粨鏋滀粎鎸夊悜閲忕浉浼煎害鎺掑簭锛屽彲鑳藉瓨鍦細

1. **璇箟婕傜Щ**: 鍚戦噺鐩镐技浣嗗疄闄呬笉鐩稿叧
2. **閬楁紡閲嶈缁撴灉**: 鐩稿叧鎬ч珮浣嗗悜閲忚窛绂昏繙
3. **缂轰箯澶氭牱鎬?*: 缁撴灉杩囦簬闆嗕腑

#### 4.2.2 鍙鎬у垎鏋?
| 缁村害 | 璇勪及 | 璇存槑 |
|-----|-----|-----|
| 鎶€鏈彲琛屾€?| 鉁?楂?| 閲嶆帓搴忕畻娉曟垚鐔?|
| 璧勬簮闇€姹?| 鈿狅笍 涓?| 闇€瑕侀澶朙LM璋冪敤 |
| 鍏煎鎬?| 鉁?楂?| 鍙彃鎷斿疄鐜?|
| 椋庨櫓 | 鉁?浣?| 涓嶅奖鍝嶇幇鏈夋祦绋?|

#### 4.2.3 鏈€浼樺疄鐜版柟妗?
**鏂规閫夋嫨**: Cross-Encoder閲嶆帓搴?+ 澶氭牱鎬ч噸鎺?
```python
# core/reranker.py (鏂版枃浠?

from typing import List, Dict, Any
from dataclasses import dataclass
import httpx
import os

@dataclass
class RerankResult:
    """閲嶆帓搴忕粨鏋?""
    id: str
    text: str
    original_score: float
    rerank_score: float
    final_score: float
    payload: Dict[str, Any]


class LLMReranker:
    """鍩轰簬LLM鐨勯噸鎺掑簭鍣?""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.api_base = "https://api.deepseek.com"
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        浣跨敤LLM瀵瑰€欓€夌粨鏋滈噸鎺掑簭
        
        绛栫暐: 璁㎜LM瀵规瘡涓€欓€夋墦鍒?        """
        if not candidates:
            return []
        
        # 鎵归噺鎵撳垎
        scored_results = []
        
        for i, candidate in enumerate(candidates[:20]):  # 鏈€澶氶噸鎺?0涓?            score = await self._score_relevance(
                query=query,
                text=candidate.get("text", ""),
                context=candidate.get("source_url", "")
            )
            
            scored_results.append(RerankResult(
                id=candidate.get("id", str(i)),
                text=candidate.get("text", ""),
                original_score=candidate.get("score", 0.0),
                rerank_score=score,
                final_score=0.0,  # 鍚庣画璁＄畻
                payload=candidate
            ))
        
        # 璁＄畻鏈€缁堝垎鏁?(鍘熷鍒嗘暟 + 閲嶆帓搴忓垎鏁扮殑鍔犳潈)
        for result in scored_results:
            result.final_score = (
                0.3 * result.original_score + 
                0.7 * result.rerank_score
            )
        
        # 鎸夋渶缁堝垎鏁版帓搴?        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_results[:top_k]
    
    async def _score_relevance(
        self,
        query: str,
        text: str,
        context: str = ""
    ) -> float:
        """浣跨敤LLM瀵瑰崟涓枃妗ｆ墦鍒?""
        
        prompt = f"""璇峰垽鏂互涓嬪唴瀹逛笌鏌ヨ鐨勭浉鍏虫€э紝骞剁粰鍑?.0鍒?.0鐨勫垎鏁般€?
鏌ヨ: {query}

鍐呭: {text[:500]}

璇勫垎鏍囧噯:
- 1.0: 鍐呭鐩存帴鍥炵瓟鏌ヨ锛屽寘鍚叧閿俊鎭?- 0.7: 鍐呭涓庢煡璇㈢浉鍏筹紝鎻愪緵鏈夌敤鑳屾櫙
- 0.4: 鍐呭闂存帴鐩稿叧锛屽彲鑳芥湁鐢?- 0.0: 鍐呭涓庢煡璇㈡棤鍏?
璇峰彧杩斿洖涓€涓暟瀛楀垎鏁帮紝涓嶈鏈夊叾浠栨枃瀛?"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0.0
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "0.5")
                score = float(content.strip())
                return max(0.0, min(1.0, score))
                
        except Exception as e:
            print(f"[LLMReranker] Error scoring: {e}")
            return 0.5


class DiversityReranker:
    """澶氭牱鎬ч噸鎺掑簭鍣?""
    
    def __init__(self, diversity_threshold: float = 0.8):
        self.diversity_threshold = diversity_threshold
    
    def rerank(
        self,
        candidates: List[RerankResult],
        embeddings: List[List[float]] = None
    ) -> List[RerankResult]:
        """
        澶氭牱鎬ч噸鎺掑簭 (MMR绠楁硶)
        
        纭繚缁撴灉澶氭牱鎬э紝閬垮厤杩囦簬鐩镐技鐨勭粨鏋?        """
        if not candidates or len(candidates) <= 1:
            return candidates
        
        selected = [candidates[0]]  # 閫夋嫨鏈€楂樺垎鐨?        remaining = candidates[1:]
        
        while remaining and len(selected) < len(candidates):
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # 璁＄畻涓庡凡閫夌粨鏋滅殑鏈€澶х浉浼煎害
                max_sim = self._max_similarity_to_selected(
                    candidate, selected, embeddings
                )
                
                # MMR鍒嗘暟 = 鐩稿叧鎬?- 位 * 鏈€澶х浉浼煎害
                mmr_score = 0.7 * candidate.final_score - 0.3 * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _max_similarity_to_selected(
        self,
        candidate: RerankResult,
        selected: List[RerankResult],
        embeddings: List[List[float]] = None
    ) -> float:
        """璁＄畻鍊欓€変笌宸查€夌粨鏋滅殑鏈€澶х浉浼煎害"""
        # 绠€鍖栧疄鐜? 鍩轰簬鏂囨湰閲嶅彔
        max_sim = 0.0
        candidate_words = set(candidate.text.lower().split())
        
        for s in selected:
            selected_words = set(s.text.lower().split())
            overlap = len(candidate_words & selected_words)
            union = len(candidate_words | selected_words)
            sim = overlap / union if union > 0 else 0.0
            max_sim = max(max_sim, sim)
        
        return max_sim


# core/rag_retriever.py 淇敼

class RAGRetriever:
    
    def __init__(self, knowledge_manager: KnowledgeManager, use_reranker: bool = True):
        self.km = knowledge_manager
        self.llm_reranker = LLMReranker() if use_reranker else None
        self.diversity_reranker = DiversityReranker()
    
    async def retrieve_for_report(self, ...) -> RetrievedContext:
        # ... 鐜版湁妫€绱㈤€昏緫
        
        # 閲嶆帓搴?        if self.llm_reranker:
            reranked = await self.llm_reranker.rerank(
                query=root_query,
                candidates=all_results,
                top_k=max_facts
            )
            
            # 澶氭牱鎬ч噸鎺?            final_results = self.diversity_reranker.rerank(reranked)
            
            all_results = [
                {
                    "id": r.id,
                    "text": r.text,
                    "score": r.final_score,
                    "original_score": r.original_score,
                    **r.payload
                }
                for r in final_results
            ]
        
        # ... 杩斿洖缁撴灉
```

#### 4.2.4 瀹炴柦姝ラ

| 姝ラ | 浠诲姟 | 棰勪及鏃堕棿 | 浜у嚭鐗?|
|-----|-----|---------|--------|
| 1 | 鍒涘缓LLMReranker绫?| 1.5h | `core/reranker.py` |
| 2 | 瀹炵幇鐩稿叧鎬ф墦鍒?| 1h | _score_relevance鏂规硶 |
| 3 | 鍒涘缓DiversityReranker | 1h | MMR绠楁硶瀹炵幇 |
| 4 | 闆嗘垚鍒癛AGRetriever | 1h | 淇敼retrieve_for_report |
| 5 | 缂栧啓娴嬭瘯 | 1h | `tests/test_reranker.py` |

#### 4.2.5 楠屾敹鏍囧噯

- [ ] LLM閲嶆帓搴忚兘鎻愬崌妫€绱㈢浉鍏虫€?- [ ] 澶氭牱鎬ч噸鎺掕兘鍑忓皯閲嶅缁撴灉
- [ ] 閲嶆帓搴忚繃绋嬫湁璇︾粏鏃ュ織
- [ ] 鏀寔閰嶇疆寮€鍏虫帶鍒舵槸鍚﹀惎鐢?- [ ] 娴嬭瘯瑕嗙洊鐜?> 80%

---

### 4.3 P1-3: 娣峰悎妫€绱?
#### 4.3.1 闂鎻忚堪

褰撳墠浠呬娇鐢ㄥ悜閲忔绱紝瀛樺湪浠ヤ笅闂锛?
1. **绮剧‘鍖归厤澶辨晥**: 鍚戦噺妫€绱㈠鍏抽敭璇嶅尮閰嶄笉鏁忔劅
2. **涓撲笟鏈妫€绱㈠樊**: 宓屽叆妯″瀷鍙兘鏃犳硶鐞嗚В涓撲笟鏈
3. **鍙洖鐜囧彈闄?*: 鍗曚竴妫€绱㈡柟寮忚鐩栭潰鏈夐檺

#### 4.3.2 鍙鎬у垎鏋?
| 缁村害 | 璇勪及 | 璇存槑 |
|-----|-----|-----|
| 鎶€鏈彲琛屾€?| 鉁?楂?| BM25绠楁硶鎴愮啛锛屾湁鐜版垚搴?|
| 璧勬簮闇€姹?| 鈿狅笍 涓?| 闇€瑕佺淮鎶ゅ€掓帓绱㈠紩 |
| 鍏煎鎬?| 鉁?楂?| 鍙笌鍚戦噺妫€绱㈠苟琛?|
| 椋庨櫓 | 鉁?浣?| 涓嶅奖鍝嶇幇鏈夋祦绋?|

#### 4.3.3 鏈€浼樺疄鐜版柟妗?
**鏂规閫夋嫨**: BM25 + 鍚戦噺妫€绱?+ RRF铻嶅悎

```python
# core/hybrid_retriever.py (鏂版枃浠?

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math
from collections import Counter

@dataclass
class HybridSearchResult:
    id: str
    text: str
    vector_score: float
    keyword_score: float
    combined_score: float
    payload: Dict[str, Any]


class BM25Index:
    """BM25鍊掓帓绱㈠紩"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.inverted_index: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: tf}
        self.doc_count: int = 0
        self.idf_cache: Dict[str, float] = {}
    
    def index(self, documents: List[Dict[str, Any]]):
        """鏋勫缓绱㈠紩"""
        self.doc_count = len(documents)
        total_length = 0
        
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # 鏋勫缓鍊掓帓绱㈠紩
            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = freq
        
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25鎼滅储"""
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._compute_idf(term)
            
            for doc_id, tf in self.inverted_index[term].items():
                doc_length = self.doc_lengths.get(doc_id, 0)
                
                # BM25鍏紡
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score = idf * numerator / denominator
                
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += score
        
        # 鎺掑簭
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _compute_idf(self, term: str) -> float:
        """璁＄畻IDF"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        df = len(self.inverted_index.get(term, {}))
        idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        
        self.idf_cache[term] = idf
        return idf
    
    def _tokenize(self, text: str) -> List[str]:
        """鍒嗚瘝 (绠€鍗曞疄鐜?"""
        import re
        # 涓枃鎸夊瓧绗﹀垎鍓诧紝鑻辨枃鎸夌┖鏍煎垎鍓?        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        english = re.findall(r'[a-zA-Z]+', text)
        
        tokens = []
        for c in chinese:
            tokens.extend(list(c))
        for e in english:
            tokens.append(e.lower())
        
        return tokens


class HybridRetriever:
    """娣峰悎妫€绱㈠櫒"""
    
    def __init__(
        self,
        knowledge_manager: 'KnowledgeManager',
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5
    ):
        self.km = knowledge_manager
        self.bm25_index = BM25Index()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._indexed = False
    
    async def build_index(self):
        """鏋勫缓娣峰悎绱㈠紩"""
        # 鑾峰彇鎵€鏈夋枃妗?        all_facts = await self.km.search_facts("", limit=1000, threshold=0.0)
        
        if all_facts:
            self.bm25_index.index(all_facts)
            self._indexed = True
            print(f"[HybridRetriever] Indexed {len(all_facts)} documents")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[HybridSearchResult]:
        """
        娣峰悎妫€绱?        
        1. 鍚戦噺妫€绱?        2. BM25妫€绱?        3. RRF铻嶅悎
        """
        if not self._indexed:
            await self.build_index()
        
        # 鍚戦噺妫€绱?        vector_results = await self.km.search_facts(
            query=query,
            limit=top_k * 2,
            threshold=0.0
        )
        vector_ranking = {r["id"]: i for i, r in enumerate(vector_results)}
        
        # BM25妫€绱?        bm25_results = self.bm25_index.search(query, top_k * 2)
        bm25_ranking = {r[0]: i for i, r in enumerate(bm25_results)}
        
        # RRF铻嶅悎
        all_doc_ids = set(vector_ranking.keys()) | set(bm25_ranking.keys())
        rrf_scores = {}
        
        k = 60  # RRF鍙傛暟
        for doc_id in all_doc_ids:
            vector_rank = vector_ranking.get(doc_id, len(vector_ranking))
            bm25_rank = bm25_ranking.get(doc_id, len(bm25_ranking))
            
            rrf_score = 1 / (k + vector_rank) + 1 / (k + bm25_rank)
            rrf_scores[doc_id] = rrf_score
        
        # 鎺掑簭骞舵瀯寤虹粨鏋?        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, combined_score in sorted_ids[:top_k]:
            # 鑾峰彇鏂囨。璇︽儏
            doc = await self.km.get_fact_by_id(doc_id)
            if doc:
                results.append(HybridSearchResult(
                    id=doc_id,
                    text=doc.get("text", ""),
                    vector_score=1 / (k + vector_ranking.get(doc_id, len(vector_ranking))),
                    keyword_score=1 / (k + bm25_ranking.get(doc_id, len(bm25_ranking))),
                    combined_score=combined_score,
                    payload=doc
                ))
        
        return results


# RRF (Reciprocal Rank Fusion) 绠楁硶璇存槑:
# score(d) = 危 1/(k + rank(d))
# 鍏朵腑 k 鏄钩婊戝弬鏁帮紝閫氬父鍙?60
```

#### 4.3.4 瀹炴柦姝ラ

| 姝ラ | 浠诲姟 | 棰勪及鏃堕棿 | 浜у嚭鐗?|
|-----|-----|---------|--------|
| 1 | 鍒涘缓BM25Index绫?| 2h | `core/hybrid_retriever.py` |
| 2 | 瀹炵幇鍊掓帓绱㈠紩鏋勫缓 | 1h | index鏂规硶 |
| 3 | 瀹炵幇BM25鎼滅储 | 1.5h | search鏂规硶 |
| 4 | 鍒涘缓HybridRetriever | 1.5h | RRF铻嶅悎 |
| 5 | 闆嗘垚鍒癛AGRetriever | 1h | 淇敼妫€绱㈡祦绋?|
| 6 | 缂栧啓娴嬭瘯 | 1h | `tests/test_hybrid_retriever.py` |

#### 4.3.5 楠屾敹鏍囧噯

- [ ] BM25绱㈠紩鑳芥纭瀯寤?- [ ] 鍏抽敭璇嶆绱㈣兘鍙洖绮剧‘鍖归厤缁撴灉
- [ ] RRF铻嶅悎鑳界患鍚堜袱绉嶆绱㈢粨鏋?- [ ] 娣峰悎妫€绱㈠彫鍥炵巼楂樹簬鍗曚竴妫€绱?- [ ] 娴嬭瘯瑕嗙洊鐜?> 80%

---

## 5. P2绾т换鍔¤瑙?
### 5.1 P2-1: 鏉ユ簮鍙俊搴﹁瘎浼?
#### 5.1.1 闂鎻忚堪

褰撳墠浜嬪疄鐨?`confidence` 浠呯敱 LLM 涓昏鍒ゆ柇锛岀己涔忓瑙傝瘎浼版爣鍑嗐€?
#### 5.1.2 鏈€浼樺疄鐜版柟妗?
```python
# core/credibility.py (鏂版枃浠?

from typing import Dict, List
from dataclasses import dataclass
import re

@dataclass
class CredibilityScore:
    overall: float
    domain_score: float
    content_score: float
    citation_score: float
    freshness_score: float
    details: Dict[str, str]


class SourceCredibilityScorer:
    """鏉ユ簮鍙俊搴﹁瘎浼板櫒"""
    
    TRUSTED_DOMAINS = {
        # 鏀垮簻鏈烘瀯
        "gov.cn": 0.95,
        "gov.uk": 0.95,
        "gov": 0.90,
        # 鏁欒偛鏈烘瀯
        "edu.cn": 0.90,
        "edu": 0.85,
        # 鏂伴椈鏈烘瀯
        "reuters.com": 0.85,
        "bloomberg.com": 0.85,
        "ft.com": 0.85,
        "wsj.com": 0.80,
        # 瀛︽湳
        "arxiv.org": 0.80,
        "nature.com": 0.90,
        "science.org": 0.90,
        # 绉戞妧鍏徃
        "openai.com": 0.75,
        "deepmind.com": 0.75,
    }
    
    SUSPICIOUS_DOMAINS = {
        "blogspot.com": 0.40,
        "wordpress.com": 0.45,
        "medium.com": 0.50,
        "substack.com": 0.50,
    }
    
    def score(self, source_url: str, content: str, published_date: str = None) -> CredibilityScore:
        """璇勪及鏉ユ簮鍙俊搴?""
        
        domain_score = self._score_domain(source_url)
        content_score = self._score_content(content)
        citation_score = self._score_citations(content)
        freshness_score = self._score_freshness(published_date)
        
        # 鍔犳潈缁煎悎鍒嗘暟
        overall = (
            0.35 * domain_score +
            0.25 * content_score +
            0.25 * citation_score +
            0.15 * freshness_score
        )
        
        return CredibilityScore(
            overall=overall,
            domain_score=domain_score,
            content_score=content_score,
            citation_score=citation_score,
            freshness_score=freshness_score,
            details={
                "domain": self._extract_domain(source_url),
                "has_data": str(self._has_numerical_data(content)),
                "has_citations": str(citation_score > 0.5),
            }
        )
    
    def _score_domain(self, url: str) -> float:
        """璇勪及鍩熷悕鍙俊搴?""
        domain = self._extract_domain(url)
        
        for trusted, score in self.TRUSTED_DOMAINS.items():
            if trusted in domain:
                return score
        
        for suspicious, score in self.SUSPICIOUS_DOMAINS.items():
            if suspicious in domain:
                return score
        
        return 0.60  # 榛樿鍒嗘暟
    
    def _score_content(self, content: str) -> float:
        """璇勪及鍐呭璐ㄩ噺"""
        score = 0.5
        
        # 鏈夋暟鍊兼暟鎹?        if self._has_numerical_data(content):
            score += 0.15
        
        # 鏈変笓涓氭湳璇?        if self._has_technical_terms(content):
            score += 0.10
        
        # 鍐呭闀垮害閫備腑
        if 200 < len(content) < 5000:
            score += 0.10
        
        # 鏈夌粨鏋勫寲鍐呭
        if "##" in content or "1." in content:
            score += 0.10
        
        # 鏃犳槑鏄惧箍鍛?        if not self._has_ads(content):
            score += 0.05
        
        return min(1.0, score)
    
    def _score_citations(self, content: str) -> float:
        """璇勪及寮曠敤璐ㄩ噺"""
        # 妫€鏌ユ槸鍚︽湁寮曠敤鏍囪
        citation_patterns = [
            r'\[\d+\]',  # [1], [2]
            r'\(.*?\d{4}.*?\)',  # (Author, 2023)
            r'鏍规嵁.*?鎶ュ憡',
            r'鏁版嵁鏄剧ず',
            r'鏉ユ簮:',
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content))
        
        if citation_count >= 3:
            return 0.90
        elif citation_count >= 1:
            return 0.70
        else:
            return 0.40
    
    def _score_freshness(self, published_date: str) -> float:
        """璇勪及鏃舵晥鎬?""
        if not published_date:
            return 0.50
        
        from datetime import datetime, timedelta
        
        try:
            pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            age = datetime.now(pub_date.tzinfo) - pub_date
            
            if age < timedelta(days=30):
                return 1.0
            elif age < timedelta(days=180):
                return 0.85
            elif age < timedelta(days=365):
                return 0.70
            else:
                return 0.50
        except:
            return 0.50
    
    def _extract_domain(self, url: str) -> str:
        """鎻愬彇鍩熷悕"""
        import re
        match = re.search(r'://([^/]+)', url)
        return match.group(1) if match else ""
    
    def _has_numerical_data(self, content: str) -> bool:
        """妫€鏌ユ槸鍚︽湁鏁板€兼暟鎹?""
        numbers = re.findall(r'\d+\.?\d*[%浜夸竾缇庡厓]?', content)
        return len(numbers) >= 3
    
    def _has_technical_terms(self, content: str) -> bool:
        """妫€鏌ユ槸鍚︽湁涓撲笟鏈"""
        terms = ['绾崇背', '鍒剁▼', 'GAA', 'FinFET', 'EUV', 'AI', 'GPU', 'CPU']
        return any(term in content for term in terms)
    
    def _has_ads(self, content: str) -> bool:
        """妫€鏌ユ槸鍚︽湁骞垮憡"""
        ad_patterns = ['骞垮憡', '璧炲姪', '鎺ㄥ箍', 'advertisement']
        return any(ad in content.lower() for ad in ad_patterns)
```

#### 5.1.3 瀹炴柦姝ラ

| 姝ラ | 浠诲姟 | 棰勪及鏃堕棿 |
|-----|-----|---------|
| 1 | 鍒涘缓SourceCredibilityScorer | 1.5h |
| 2 | 闆嗘垚鍒癉istillerAgent | 1h |
| 3 | 鏇存柊confidence璁＄畻閫昏緫 | 1h |
| 4 | 缂栧啓娴嬭瘯 | 1h |

---

### 5.2 P2-2: 鐗堟湰绠＄悊

#### 5.2.1 闂鎻忚堪

浜嬪疄娌℃湁鐗堟湰鍘嗗彶锛屾棤娉曡拷婧彉鏇达紝鏃犳硶鍥炴粴銆?
#### 5.2.2 鏈€浼樺疄鐜版柟妗?
```python
# core/version_manager.py (鏂版枃浠?

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class FactVersion:
    version: int
    text: str
    confidence: float
    source_url: str
    changed_at: datetime
    change_reason: str
    previous_version: Optional[int] = None


class FactVersionManager:
    """浜嬪疄鐗堟湰绠＄悊鍣?""
    
    def __init__(self, storage_path: str = "./fact_versions"):
        self.storage_path = storage_path
        self._versions: Dict[str, List[FactVersion]] = {}
    
    def create_version(
        self,
        fact_id: str,
        text: str,
        confidence: float,
        source_url: str,
        reason: str = "initial"
    ) -> FactVersion:
        """鍒涘缓鏂扮増鏈?""
        existing = self._versions.get(fact_id, [])
        
        new_version = FactVersion(
            version=len(existing) + 1,
            text=text,
            confidence=confidence,
            source_url=source_url,
            changed_at=datetime.now(),
            change_reason=reason,
            previous_version=existing[-1].version if existing else None
        )
        
        if fact_id not in self._versions:
            self._versions[fact_id] = []
        self._versions[fact_id].append(new_version)
        
        return new_version
    
    def get_version(self, fact_id: str, version: int = None) -> Optional[FactVersion]:
        """鑾峰彇鎸囧畾鐗堟湰"""
        versions = self._versions.get(fact_id, [])
        if not versions:
            return None
        
        if version is None:
            return versions[-1]
        
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def get_history(self, fact_id: str) -> List[FactVersion]:
        """鑾峰彇鐗堟湰鍘嗗彶"""
        return self._versions.get(fact_id, [])
    
    def rollback(self, fact_id: str, target_version: int) -> Optional[FactVersion]:
        """鍥炴粴鍒版寚瀹氱増鏈?""
        target = self.get_version(fact_id, target_version)
        if target:
            # 鍒涘缓涓€涓柊鐗堟湰锛屽唴瀹规槸鐩爣鐗堟湰鐨勫唴瀹?            return self.create_version(
                fact_id=fact_id,
                text=target.text,
                confidence=target.confidence,
                source_url=target.source_url,
                reason=f"rollback to version {target_version}"
            )
        return None
```

#### 5.2.3 瀹炴柦姝ラ

| 姝ラ | 浠诲姟 | 棰勪及鏃堕棿 |
|-----|-----|---------|
| 1 | 鍒涘缓FactVersionManager | 1.5h |
| 2 | 闆嗘垚鍒癒nowledgeManager | 1h |
| 3 | 娣诲姞鐗堟湰鍘嗗彶API | 1h |
| 4 | 缂栧啓娴嬭瘯 | 1h |

---

## 6. 瀹炴柦璺嚎鍥?
### 6.1 闃舵鍒掑垎

```
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?                        瀹炴柦璺嚎鍥?                                   鈹?鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?鈹?                                                                    鈹?鈹? 闃舵1: 鍩虹璁炬柦 (P0) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€  鈹?鈹? 鈹?                                                                 鈹?鈹? 鈹溾攢鈹€ P0-1: 鍚戦噺鏁版嵁搴撻泦鎴?                                          鈹?鈹? 鈹?  鈹斺攢鈹€ 浜у嚭: VectorStoreAdapter, KnowledgeManager閲嶆瀯             鈹?鈹? 鈹?                                                                 鈹?鈹? 鈹斺攢鈹€ P0-2: RAG妫€绱㈠疄鐜?                                             鈹?鈹?     鈹斺攢鈹€ 浜у嚭: RAGRetriever, Writer鏀归€?                            鈹?鈹?                                                                    鈹?鈹? 闃舵2: 鏁堟灉鎻愬崌 (P1) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€  鈹?鈹? 鈹?                                                                 鈹?鈹? 鈹溾攢鈹€ P1-1: 鏂囨。鍒嗗潡绛栫暐                                             鈹?鈹? 鈹?  鈹斺攢鈹€ 浜у嚭: SemanticChunker, DistillerAgent鏀归€?                 鈹?鈹? 鈹?                                                                 鈹?鈹? 鈹溾攢鈹€ P1-2: 閲嶆帓搴忔満鍒?                                              鈹?鈹? 鈹?  鈹斺攢鈹€ 浜у嚭: LLMReranker, DiversityReranker                       鈹?鈹? 鈹?                                                                 鈹?鈹? 鈹斺攢鈹€ P1-3: 娣峰悎妫€绱?                                                鈹?鈹?     鈹斺攢鈹€ 浜у嚭: BM25Index, HybridRetriever                           鈹?鈹?                                                                    鈹?鈹? 闃舵3: 璐ㄩ噺淇濋殰 (P2) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€  鈹?鈹? 鈹?                                                                 鈹?鈹? 鈹溾攢鈹€ P2-1: 鏉ユ簮鍙俊搴﹁瘎浼?                                          鈹?鈹? 鈹?  鈹斺攢鈹€ 浜у嚭: SourceCredibilityScorer                              鈹?鈹? 鈹?                                                                 鈹?鈹? 鈹斺攢鈹€ P2-2: 鐗堟湰绠＄悊                                                 鈹?鈹?     鈹斺攢鈹€ 浜у嚭: FactVersionManager                                   鈹?鈹?                                                                    鈹?鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

### 6.2 鏃堕棿瑙勫垝

| 闃舵 | 浠诲姟 | 棰勪及鏃堕棿 | 渚濊禆 |
|-----|-----|---------|-----|
| **闃舵1** | P0-1 鍚戦噺鏁版嵁搴撻泦鎴?| 6.5h | 鏃?|
| | P0-2 RAG妫€绱㈠疄鐜?| 5h | P0-1 |
| **闃舵2** | P1-1 鏂囨。鍒嗗潡绛栫暐 | 8h | 鏃?|
| | P1-2 閲嶆帓搴忔満鍒?| 5.5h | P0-2 |
| | P1-3 娣峰悎妫€绱?| 8h | P0-1 |
| **闃舵3** | P2-1 鏉ユ簮鍙俊搴﹁瘎浼?| 4.5h | 鏃?|
| | P2-2 鐗堟湰绠＄悊 | 4.5h | 鏃?|
| **鎬昏** | | **42h** | |

### 6.3 閲岀▼纰?
| 閲岀▼纰?| 瀹屾垚鏍囧噯 | 棰勮瀹屾垚 |
|-------|---------|---------|
| M1 | 鍚戦噺鏁版嵁搴撻泦鎴愬畬鎴愶紝娴嬭瘯閫氳繃 | 闃舵1寮€濮嬪悗8h |
| M2 | RAG妫€绱㈠彲鐢紝鎶ュ憡璐ㄩ噺鎻愬崌 | 闃舵1寮€濮嬪悗13h |
| M3 | 鍒嗗潡绛栫暐涓婄嚎锛屼簨瀹炴彁鍙栬川閲忔彁鍗?| 闃舵2寮€濮嬪悗8h |
| M4 | 閲嶆帓搴?娣峰悎妫€绱㈠畬鎴愶紝妫€绱㈡晥鏋滄彁鍗?| 闃舵2寮€濮嬪悗21.5h |
| M5 | 鍏ㄩ儴P0-P2浠诲姟瀹屾垚 | 闃舵3寮€濮嬪悗9h |

---

## 7. 椋庨櫓璇勪及涓庡簲瀵?
### 7.1 鎶€鏈闄?
| 椋庨櫓 | 鍙兘鎬?| 褰卞搷 | 搴斿鎺柦 |
|-----|-------|-----|---------|
| Qdrant閮ㄧ讲澶辫触 | 浣?| 楂?| 鎻愪緵鍐呭瓨瀛樺偍浣滀负fallback |
| LLM API闄愭祦 | 涓?| 涓?| 瀹炵幇璇锋眰闃熷垪鍜岄噸璇曟満鍒?|
| 鍒嗗潡瀵艰嚧浜嬪疄鏂 | 涓?| 涓?| 娣诲姞閲嶅彔绐楀彛鍜岃法鍧楀悎骞?|
| 閲嶆帓搴忓欢杩熻繃楂?| 涓?| 浣?| 鏀寔閰嶇疆寮€鍏筹紝鍙鐢?|

### 7.2 璧勬簮椋庨櫓

| 椋庨櫓 | 鍙兘鎬?| 褰卞搷 | 搴斿鎺柦 |
|-----|-------|-----|---------|
| API鎴愭湰澧炲姞 | 楂?| 涓?| 瀹炵幇缂撳瓨鏈哄埗锛屼紭鍖栬皟鐢ㄦ鏁?|
| 瀛樺偍绌洪棿涓嶈冻 | 浣?| 涓?| 瀹炵幇鏁版嵁娓呯悊绛栫暐 |
| 鍐呭瓨鍗犵敤杩囬珮 | 涓?| 涓?| 浣跨敤Qdrant鏇夸唬鍐呭瓨瀛樺偍 |

### 7.3 鍏煎鎬ч闄?
| 椋庨櫓 | 鍙兘鎬?| 褰卞搷 | 搴斿鎺柦 |
|-----|-------|-----|---------|
| 鐜版湁娴嬭瘯澶辫触 | 涓?| 楂?| 淇濇寔鍚戝悗鍏煎锛岄€愭杩佺Щ |
| 閰嶇疆杩佺Щ闂 | 浣?| 涓?| 鎻愪緵杩佺Щ鑴氭湰鍜屾枃妗?|

---

## 闄勫綍

### A. 鏂囦欢鍙樻洿娓呭崟

| 鏂囦欢 | 鎿嶄綔 | 璇存槑 |
|-----|-----|-----|
| `core/vector_store_adapter.py` | 鏂板 | 鍚戦噺瀛樺偍閫傞厤鍣ㄦ帴鍙?|
| `core/knowledge.py` | 淇敼 | 閲嶆瀯浠ユ敮鎸佸绉嶅瓨鍌ㄥ悗绔?|
| `core/rag_retriever.py` | 鏂板 | RAG妫€绱㈠櫒 |
| `core/chunker.py` | 鏂板 | 璇箟鍒嗗潡鍣?|
| `core/reranker.py` | 鏂板 | 閲嶆帓搴忓櫒 |
| `core/hybrid_retriever.py` | 鏂板 | 娣峰悎妫€绱㈠櫒 |
| `core/credibility.py` | 鏂板 | 鍙俊搴﹁瘎浼板櫒 |
| `core/version_manager.py` | 鏂板 | 鐗堟湰绠＄悊鍣?|
| `agents/distiller.py` | 淇敼 | 闆嗘垚鍒嗗潡绛栫暐 |
| `core/graph.py` | 淇敼 | 闆嗘垚RAG妫€绱?|
| `core/config.py` | 淇敼 | 娣诲姞鏂伴厤缃」 |

### B. 閰嶇疆椤规竻鍗?
```python
# core/config.py 鏂板閰嶇疆

class RAGConfig:
    # 鍚戦噺妫€绱?    VECTOR_SEARCH_LIMIT: int = 20
    VECTOR_SCORE_THRESHOLD: float = 0.3
    
    # 閲嶆帓搴?    ENABLE_RERANKER: bool = True
    RERANK_TOP_K: int = 10
    DIVERSITY_THRESHOLD: float = 0.8
    
    # 娣峰悎妫€绱?    ENABLE_HYBRID_SEARCH: bool = True
    VECTOR_WEIGHT: float = 0.5
    KEYWORD_WEIGHT: float = 0.5
    
    # 鍒嗗潡
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50
    CHUNK_MIN_TOKENS: int = 100
    
    # 鍙俊搴?    ENABLE_CREDIBILITY_SCORING: bool = True
    MIN_CREDIBILITY_THRESHOLD: float = 0.5
```

### C. 娴嬭瘯娓呭崟

| 娴嬭瘯鏂囦欢 | 娴嬭瘯鍐呭 |
|---------|---------|
| `tests/test_vector_adapter.py` | 鍚戦噺瀛樺偍閫傞厤鍣?|
| `tests/test_rag_retriever.py` | RAG妫€绱㈠櫒 |
| `tests/test_chunker.py` | 璇箟鍒嗗潡鍣?|
| `tests/test_reranker.py` | 閲嶆帓搴忓櫒 |
| `tests/test_hybrid_retriever.py` | 娣峰悎妫€绱㈠櫒 |
| `tests/test_credibility.py` | 鍙俊搴﹁瘎浼板櫒 |
| `tests/test_version_manager.py` | 鐗堟湰绠＄悊鍣?|

---

> 鏂囨。缁撴潫  
> 涓嬩竴姝? 鎸夌収浼樺厛绾ч『搴忓紑濮嬪疄鏂?P0-1 浠诲姟

