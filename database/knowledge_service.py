"""
知识库服务
=============

提供知识库的CRUD操作和检索功能。
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session
import jieba
import re
from utils.logger_loguru import get_logger
from database.models import (
    Base, ProductKnowledge, CustomerServiceKnowledge, KnowledgeMetaEntry, Shop,
    PresaleKnowledge, InsaleKnowledge, AftersaleKnowledge,
)
from database.db_manager import db_manager
from database.vector_retriever import VectorItem, VectorRetriever

logger = get_logger("KnowledgeService")
MIN_PRODUCT_HIT_CHARS = 80
CUSTOMER_SCENE_LABELS = {
    "presale": "售前",
    "insale": "售中",
    "aftersale": "售后",
}
CUSTOMER_SCENE_ALIASES = {
    "presale": (
        "售前", "售前咨询", "购买前", "下单前", "拍前", "买前", "购买咨询",
        "pre_sale", "presale", "pre-sale",
    ),
    "insale": (
        "售中", "售中-待发货", "售中-物流中", "待发货", "已发货待收货",
        "物流中", "催发货", "加急发货", "改地址", "修改地址", "拦截",
        "insale", "in_sale", "in-sale",
    ),
    "aftersale": (
        "售后", "售后倾向", "已签收", "签收后", "收到后", "质量问题",
        "退换货", "退货退款", "退款补偿", "协商", "aftersale", "after_sale",
        "after-sale",
    ),
}


class KnowledgeService:
    """知识库服务，提供产品知识和客服知识的CRUD和检索功能"""

    def __init__(self):
        """初始化知识库服务"""
        # 复用现有的数据库管理器，确保路径一致
        self.session_factory = db_manager.Session
        self.vector_retriever = VectorRetriever()
        # 确保知识库相关的表存在
        Base.metadata.create_all(db_manager.engine)
        logger.info("KnowledgeService 初始化成功，复用全局数据库连接")

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.session_factory()

    # ========== Meta 知识 ==========

    def replace_meta_entries(
        self,
        shop_id: int,
        entries: List[Dict[str, Any]],
        source_type: Optional[str] = None,
        product_family: Optional[str] = None,
    ) -> int:
        """按来源批量重建 meta 知识。"""
        with self.get_session() as session:
            conditions = [KnowledgeMetaEntry.shop_id == shop_id]
            if source_type:
                conditions.append(KnowledgeMetaEntry.source_type == source_type)
            if product_family:
                conditions.append(KnowledgeMetaEntry.product_family == product_family)

            session.query(KnowledgeMetaEntry).filter(and_(*conditions)).delete()

            now = datetime.now()
            created = 0
            for item in entries:
                meta = KnowledgeMetaEntry(
                    shop_id=shop_id,
                    source_type=item["source_type"],
                    source_id=int(item["source_id"]),
                    goods_id=item.get("goods_id"),
                    product_family=item.get("product_family"),
                    scenario=item["scenario"],
                    sub_intent=item.get("sub_intent"),
                    aliases=item["aliases"],
                    answer=item["answer"],
                    section_title=item.get("section_title"),
                    tags=item.get("tags"),
                    enabled=bool(item.get("enabled", True)),
                    priority=int(item.get("priority", 0)),
                    created_at=now,
                    updated_at=now,
                )
                session.add(meta)
                created += 1
            session.commit()
            logger.info(
                f"Meta知识重建完成: shop_id={shop_id}, source_type={source_type}, "
                f"product_family={product_family}, created={created}"
            )
            return created

    # ========== 产品知识 ==========

    def get_product_by_goods_id(self, shop_id: int, goods_id: int) -> Optional[ProductKnowledge]:
        """根据商品ID获取产品知识"""
        with self.get_session() as session:
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            return session.scalar(stmt)

    def list_products_by_shop(self, shop_id: int) -> List[ProductKnowledge]:
        """获取店铺所有产品知识"""
        with self.get_session() as session:
            stmt = select(ProductKnowledge).where(
                ProductKnowledge.shop_id == shop_id
            ).order_by(
                ProductKnowledge.last_extracted_at.desc(),
                ProductKnowledge.updated_at.desc(),
                ProductKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def count_products_by_shop(self, shop_id: int) -> int:
        """统计店铺产品知识数量"""
        with self.get_session() as session:
            return session.query(ProductKnowledge).filter(
                ProductKnowledge.shop_id == shop_id
            ).count()

    def add_or_update_product(
        self,
        shop_id: int,
        goods_id: int,
        goods_name: str,
        price: Optional[str] = None,
        price_min: Optional[int] = None,
        price_max: Optional[int] = None,
        sold_quantity: Optional[int] = None,
        thumb_url: Optional[str] = None,
        specifications: Optional[str] = None,
        extracted_content: Optional[str] = None,
    ) -> ProductKnowledge:
        """添加或更新产品知识"""
        with self.get_session() as session:
            # 在同一个 session 中查询
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            existing = session.scalar(stmt)

            if existing:
                # 更新现有记录
                if goods_name is not None:
                    existing.goods_name = goods_name
                if price is not None:
                    existing.price = price
                if price_min is not None:
                    existing.price_min = price_min
                if price_max is not None:
                    existing.price_max = price_max
                if sold_quantity is not None:
                    existing.sold_quantity = sold_quantity
                if thumb_url is not None:
                    existing.thumb_url = thumb_url
                if specifications is not None:
                    existing.specifications = specifications
                if extracted_content is not None:
                    existing.extracted_content = extracted_content
                existing.last_extracted_at = datetime.now()
                product = existing
                session.flush()
            else:
                # 创建新记录
                product = ProductKnowledge(
                    shop_id=shop_id,
                    goods_id=goods_id,
                    goods_name=goods_name,
                    price=price,
                    price_min=price_min,
                    price_max=price_max,
                    sold_quantity=sold_quantity,
                    thumb_url=thumb_url,
                    specifications=specifications,
                    extracted_content=extracted_content,
                )
                session.add(product)
                session.flush()

            session.commit()
            # 重新查询以确保返回的是附加到 session 的对象
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            result = session.scalar(stmt)
            logger.info(f"产品知识保存成功: shop_id={shop_id}, goods_id={goods_id}")
            return result

    def update_product_extracted_content(
        self,
        shop_id: int,
        goods_id: int,
        specifications: Optional[str] = None,
        extracted_content: Optional[str] = None,
    ) -> bool:
        """仅更新产品的提取内容（用于第二阶段更新）"""
        with self.get_session() as session:
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            product = session.scalar(stmt)
            if not product:
                logger.warning(f"产品不存在，无法更新提取内容: shop_id={shop_id}, goods_id={goods_id}")
                return False

            if specifications is not None:
                product.specifications = specifications
            if extracted_content is not None:
                product.extracted_content = extracted_content
            product.last_extracted_at = datetime.now()

            session.commit()
            logger.info(f"产品提取内容更新成功: shop_id={shop_id}, goods_id={goods_id}")
            return True

    def delete_product(self, product_id: int) -> bool:
        """删除产品知识"""
        with self.get_session() as session:
            product = session.get(ProductKnowledge, product_id)
            if not product:
                return False
            session.delete(product)
            session.commit()
            logger.info(f"产品知识删除成功: id={product_id}")
            return True

    def clear_products_by_shop(self, shop_id: int) -> int:
        """清空店铺所有产品知识，返回删除数量"""
        with self.get_session() as session:
            count = session.query(ProductKnowledge).filter(
                ProductKnowledge.shop_id == shop_id
            ).delete()
            session.commit()
            logger.info(f"清空店铺产品知识: shop_id={shop_id}, deleted={count}")
            return count

    # ========== 客服知识 ==========

    def get_customer_service_by_id(self, cs_id: int) -> Optional[CustomerServiceKnowledge]:
        """根据ID获取客服知识"""
        with self.get_session() as session:
            return session.get(CustomerServiceKnowledge, cs_id)

    def list_customer_service_by_shop(self, shop_id: int) -> List[CustomerServiceKnowledge]:
        """获取店铺所有启用的客服知识"""
        with self.get_session() as session:
            stmt = select(CustomerServiceKnowledge).where(
                and_(
                    CustomerServiceKnowledge.shop_id == shop_id,
                    CustomerServiceKnowledge.enabled == True
                )
            ).order_by(
                CustomerServiceKnowledge.updated_at.desc(),
                CustomerServiceKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def list_customer_service_with_disabled(self, shop_id: int) -> List[CustomerServiceKnowledge]:
        """获取店铺所有客服知识（包括禁用的）"""
        with self.get_session() as session:
            stmt = select(CustomerServiceKnowledge).where(
                CustomerServiceKnowledge.shop_id == shop_id
            ).order_by(
                CustomerServiceKnowledge.updated_at.desc(),
                CustomerServiceKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def count_customer_service_by_shop(self, shop_id: int) -> int:
        """统计店铺客服知识数量"""
        with self.get_session() as session:
            return session.query(CustomerServiceKnowledge).filter(
                CustomerServiceKnowledge.shop_id == shop_id
            ).count()

    def add_customer_service(
        self,
        shop_id: int,
        title: str,
        content: str,
        tags: Optional[str] = None,
        enabled: bool = True,
    ) -> CustomerServiceKnowledge:
        """添加客服知识"""
        with self.get_session() as session:
            cs = CustomerServiceKnowledge(
                shop_id=shop_id,
                title=title,
                content=content,
                tags=tags,
                enabled=enabled,
            )
            session.add(cs)
            session.commit()
            logger.info(f"客服知识添加成功: shop_id={shop_id}, title={title}")
            return cs

    def update_customer_service(
        self,
        cs_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[CustomerServiceKnowledge]:
        """更新客服知识"""
        with self.get_session() as session:
            cs = session.get(CustomerServiceKnowledge, cs_id)
            if not cs:
                return None
            if title is not None:
                cs.title = title
            if content is not None:
                cs.content = content
            if tags is not None:
                cs.tags = tags
            if enabled is not None:
                cs.enabled = enabled
            session.commit()
            logger.info(f"客服知识更新成功: id={cs_id}")
            return cs

    def delete_customer_service(self, cs_id: int) -> bool:
        """删除客服知识"""
        with self.get_session() as session:
            cs = session.get(CustomerServiceKnowledge, cs_id)
            if not cs:
                return False
            session.delete(cs)
            session.commit()
            logger.info(f"客服知识删除成功: id={cs_id}")
            return True

    def batch_import_customer_service(
        self,
        shop_id: int,
        rows: List[Dict[str, Any]],
    ) -> tuple[int, int]:
        """批量导入客服知识，跳过重复项（同店铺内标题+内容完全相同）

        Args:
            shop_id: 店铺数据库ID
            rows: 待导入行列表，每项含 title, content, tags

        Returns:
            (success_count, skipped_count)
        """
        success = 0
        skipped = 0
        with self.get_session() as session:
            for row in rows:
                title = row.get("title", "")
                content = row.get("content", "")
                tags = row.get("tags")

                # 重复检测：同店铺下标题+内容完全相同
                stmt = select(CustomerServiceKnowledge).where(
                    and_(
                        CustomerServiceKnowledge.shop_id == shop_id,
                        CustomerServiceKnowledge.title == title,
                        CustomerServiceKnowledge.content == content,
                    )
                )
                if session.scalar(stmt) is not None:
                    skipped += 1
                    continue

                cs = CustomerServiceKnowledge(
                    shop_id=shop_id,
                    title=title,
                    content=content,
                    tags=tags,
                    enabled=True,
                )
                session.add(cs)
                success += 1

            session.commit()
        logger.info(f"批量导入客服知识: shop_id={shop_id}, success={success}, skipped={skipped}")
        return success, skipped

    def filter_customer_service_by_tag(self, shop_id: int, tag: str) -> List[CustomerServiceKnowledge]:
        """按标签筛选客服知识"""
        with self.get_session() as session:
            # LIKE 查询匹配标签
            stmt = select(CustomerServiceKnowledge).where(
                and_(
                    CustomerServiceKnowledge.shop_id == shop_id,
                    CustomerServiceKnowledge.enabled == True,
                    CustomerServiceKnowledge.tags.like(f"%{tag}%"),
                )
            ).order_by(
                CustomerServiceKnowledge.updated_at.desc(),
                CustomerServiceKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def get_all_tags(self, shop_id: int) -> List[str]:
        """获取店铺所有标签（去重）"""
        with self.get_session() as session:
            stmt = select(CustomerServiceKnowledge.tags).where(
                CustomerServiceKnowledge.shop_id == shop_id
            )
            tags_list = []
            for row in session.execute(stmt):
                if row[0]:
                    tags_list.extend([t.strip() for t in row[0].split(',') if t.strip()])
            # 去重
            return sorted(list(set(tags_list)))

    # ========== 检索 ==========

    def _resolve_shop_id(self, shop_id: int) -> int:
        """
        将店铺原始ID转换为数据库中的Shop.id

        Args:
            shop_id: 店铺原始ID（如591119888）

        Returns:
            数据库中的Shop.id（如1），如果找不到返回原值
        """
        with self.get_session() as session:
            stmt = select(Shop).where(Shop.shop_id == str(shop_id))
            shop = session.scalar(stmt)
            if shop:
                return shop.id
            # 如果没找到，尝试直接用整数查询（兼容已有数据）
            stmt2 = select(Shop).where(Shop.id == shop_id)
            shop2 = session.scalar(stmt2)
            if shop2:
                return shop2.id
            # 找不到时返回原值，让后续查询返回空结果
            logger.warning(f"未找到店铺: shop_id={shop_id}")
            return shop_id

    def _product_vector_items(self, products: List[ProductKnowledge]) -> List[VectorItem]:
        items = []
        for product in products:
            if not (product.extracted_content or "").strip():
                continue
            parts = [
                product.goods_name or "",
                product.price or "",
                product.specifications or "",
                product.extracted_content or "",
            ]
            text = "\n".join(part for part in parts if part)
            if text.strip():
                items.append(VectorItem(f"product:{product.id}", text, product))
        return items

    def _customer_service_vector_items(self, cs_list: List[CustomerServiceKnowledge]) -> List[VectorItem]:
        items = []
        for cs in cs_list:
            parts = [
                cs.title or "",
                cs.tags or "",
                cs.content or "",
            ]
            text = "\n".join(part for part in parts if part)
            if text.strip():
                items.append(VectorItem(f"customer_service:{cs.id}", text, cs))
        return items

    def _product_content_vector_items(self, product: ProductKnowledge) -> List[VectorItem]:
        content = product.extracted_content or ""
        chunks = self._product_knowledge_blocks(content) or self._chunk_text(content, max_chars=420, overlap_chars=80)
        return [
            VectorItem(
                f"product:{product.id}:chunk:{index}",
                "\n".join(part for part in [product.goods_name or "", chunk] if part),
                self._strip_embedding_aliases(chunk),
            )
            for index, chunk in enumerate(chunks)
            if chunk.strip()
        ]

    def _rank_product_content(self, product: ProductKnowledge, query: Optional[str], limit: int = 5) -> str:
        if not query or not query.strip() or not (product.extracted_content or "").strip():
            return ""

        structured_content = self._rank_product_faq_content(product.extracted_content or "", query)
        if structured_content:
            return structured_content

        vector_chunks = self.vector_retriever.rank(
            namespace=f"product_knowledge_{product.id}_chunks",
            shop_id=product.shop_id,
            query=query,
            items=self._product_content_vector_items(product),
            limit=limit,
        )
        keyword_content = self._keyword_product_content(product.extracted_content or "", query, limit)
        if vector_chunks:
            return self._merge_text_blocks(
                keyword_content.split("\n\n") if keyword_content else [],
                [str(chunk) for chunk in vector_chunks],
                limit,
            )

        return keyword_content

    @classmethod
    def _rank_product_faq_content(cls, content: str, query: str) -> str:
        records = cls._product_faq_records(content)
        if not records:
            return ""

        scored = []
        for index, record in enumerate(records):
            score = cls._structured_match_score(
                query=query,
                aliases=record["aliases"],
                answer=record["answer"],
                section=record["section"],
            )
            if score > 0:
                scored.append((score, index, record))

        if not scored:
            return ""

        scored.sort(key=lambda item: (-item[0], item[1]))
        top_score, _, top_record = scored[0]
        if top_score < 8:
            return ""

        return (
            f"{top_record['section']}\n"
            f"{top_record['answer']}"
        ).strip()

    @classmethod
    def _product_faq_records(cls, content: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        records: List[Dict[str, str]] = []
        section = ""
        index = 0

        while index < len(lines):
            line = lines[index]
            if line.startswith("###"):
                section = line
                index += 1
                continue
            if line.startswith("##"):
                section = line
                index += 1
                continue

            aliases = cls._label_value(line, ("问法",))
            if not aliases:
                index += 1
                continue

            answer = ""
            next_index = index + 1
            while next_index < len(lines):
                next_line = lines[next_index]
                if next_line.startswith(("##", "###")) or cls._label_value(next_line, ("问法",)):
                    break
                answer = cls._label_value(next_line, ("要点", "答案"))
                if answer:
                    break
                next_index += 1

            if answer:
                records.append({
                    "section": section,
                    "aliases": aliases,
                    "answer": answer,
                })
                index = next_index + 1
                continue

            index += 1

        return records

    @staticmethod
    def _label_value(line: str, labels: tuple[str, ...]) -> str:
        clean = str(line or "").strip()
        if clean.startswith("-"):
            clean = clean[1:].strip()
        for label in labels:
            for separator in ("：", ":"):
                prefix = f"{label}{separator}"
                if clean.startswith(prefix):
                    return clean[len(prefix):].strip()
        return ""

    @classmethod
    def _structured_match_score(
        cls,
        query: str,
        aliases: str,
        answer: str,
        section: str = "",
        tags: str = "",
    ) -> int:
        query_clean = cls._normalize_query_match_text(query)
        aliases_clean = cls._normalize_match_text(aliases)
        answer_clean = cls._normalize_match_text(answer)
        section_clean = cls._normalize_match_text(section)
        tags_clean = cls._normalize_match_text(tags)

        score = 0
        best_direct_alias_score = 0
        weak_alias_score = 0
        for alias in re.split(r"[/|;；\n\r]+", aliases or ""):
            alias_clean = cls._normalize_match_text(alias)
            if len(alias_clean) < 2:
                continue
            alias_score = 0
            if alias_clean == query_clean:
                alias_score = 240 + min(len(alias_clean), 30)
            elif len(alias_clean) >= 4 and alias_clean in query_clean:
                alias_score = 115 + min(len(alias_clean), 12)
            elif len(query_clean) >= 6 and query_clean in alias_clean:
                alias_score = 80 + min(len(query_clean), 12)

            if alias_score:
                best_direct_alias_score = max(best_direct_alias_score, alias_score)
            elif alias_clean in query_clean or query_clean in alias_clean:
                weak_alias_score = max(weak_alias_score, 8 + min(len(alias_clean), 6))
        score += best_direct_alias_score + weak_alias_score

        for number in re.findall(r"\d+", query_clean):
            if len(number) < 2:
                continue
            if number in aliases_clean:
                score += 45
            elif number in answer_clean or number in section_clean or number in tags_clean:
                score += 18

        for term in cls._search_terms(query):
            term_clean = cls._normalize_match_text(term)
            if not term_clean:
                continue
            if term_clean in aliases_clean:
                score += 8
            if term_clean in section_clean:
                score += 4
            if term_clean in answer_clean:
                score += 2
            if term_clean in tags_clean:
                score += 2

        direct_alias_matched = best_direct_alias_score > 0
        query_scenario = cls._detect_query_scenario(query)
        scenario_anchors = cls._scenario_anchor_terms(query_scenario)
        if scenario_anchors:
            if any(term in section_clean or term in tags_clean for term in scenario_anchors):
                score += 30
            elif any(term in aliases_clean or term in answer_clean for term in scenario_anchors):
                score += 8

        query_intent = cls._detect_query_intent(query)
        score += cls._intent_specific_match_score(
            query_clean=query_clean,
            aliases_clean=aliases_clean,
            answer_clean=answer_clean,
            section_clean=section_clean,
            tags_clean=tags_clean,
        )
        if query_intent == "charge_abnormal":
            abnormal_terms = tuple(
                cls._normalize_match_text(term)
                for term in (
                    "充电异常", "充不了", "充不进", "充不上", "不能充电",
                    "无法充电", "没法充电", "充电没反应", "充电不亮",
                )
            )
            non_fault_charge_terms = tuple(
                cls._normalize_match_text(term)
                for term in (
                    "电量显示", "不插电", "怎么看电量", "充电时间", "充满",
                    "充电线", "充电器", "充电头", "边充边用", "充电口", "指示灯",
                )
            )
            structured_text = f"{section_clean}{tags_clean}{aliases_clean}"
            if any(term in structured_text for term in abnormal_terms):
                score += 60
            elif any(term in structured_text for term in non_fault_charge_terms):
                score -= 40

        for qualifier_group in cls._qualifier_groups():
            query_has = any(term in query_clean for term in qualifier_group)
            knowledge_has = any(term in aliases_clean or term in answer_clean for term in qualifier_group)
            if query_has and knowledge_has:
                score += 14
            elif not direct_alias_matched:
                if query_has and not knowledge_has:
                    score -= 18
                elif knowledge_has and not query_has:
                    score -= 18

        knowledge_scenario = cls._detect_query_scenario(f"{section} {tags} {aliases} {answer}")
        if query_scenario and knowledge_scenario:
            if query_scenario == knowledge_scenario:
                score += 36
            elif not direct_alias_matched:
                score -= 42

        knowledge_intent = cls._detect_query_intent(f"{aliases} {answer}")
        if query_intent and knowledge_intent:
            if query_intent == knowledge_intent:
                score += 24
            elif not direct_alias_matched:
                score -= 24
                hard_conflict_intents = {
                    "shipping_time", "shipping_express", "shipping_origin",
                    "return_policy", "return_shipping", "warranty",
                }
                if query_intent in hard_conflict_intents or knowledge_intent in hard_conflict_intents:
                    score -= 36

        return score

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        return re.sub(r"[\s\?？!！,，。.;；:：、~～\[\]【】()（）]+", "", str(text or "").lower())

    @classmethod
    def _normalize_query_match_text(cls, text: str) -> str:
        clean = cls._normalize_match_text(text)
        for prefix in ("内容客户消息", "客户消息"):
            if clean.startswith(prefix):
                return clean[len(prefix):]
        return clean

    @classmethod
    def _has_exact_alias_match(cls, query: str, aliases: str) -> bool:
        query_clean = cls._normalize_query_match_text(query)
        if not query_clean:
            return False
        return any(
            cls._normalize_match_text(alias) == query_clean
            for alias in re.split(r"[/|;；\n\r]+", aliases or "")
        )

    @classmethod
    def _scenario_anchor_terms(cls, scenario: str) -> tuple[str, ...]:
        anchors = {
            "charge_power": ("充电", "充电异常", "充电用电"),
            "color_purchase": ("颜色", "库存", "有货", "购买相关"),
            "product_usage": ("使用", "开关", "挂脖", "挂绳", "产品使用"),
            "cooling": ("制冷", "冰敷", "凉感"),
            "noise": ("噪音", "声音", "静音", "分贝", "风力噪音"),
            "shipping": ("发货", "物流", "快递"),
            "aftersale": ("售后", "退货", "退款", "质保", "质量问题"),
            "battery_endurance": ("续航", "电池", "充满"),
            "wind_power": ("风力", "风速", "档位"),
            "size_weight": ("尺寸", "重量", "大小"),
        }
        return tuple(cls._normalize_match_text(term) for term in anchors.get(scenario or "", ()))

    @classmethod
    def _intent_specific_match_score(
        cls,
        query_clean: str,
        aliases_clean: str,
        answer_clean: str,
        section_clean: str,
        tags_clean: str,
    ) -> int:
        """处理短问法和高风险相近主题，避免旧泛答案抢过精细知识。"""
        score = 0
        structured_text = f"{section_clean}{tags_clean}{aliases_clean}{answer_clean}"

        def q_has(*terms: str) -> bool:
            return any(cls._normalize_match_text(term) in query_clean for term in terms)

        def k_has(*terms: str) -> bool:
            return any(cls._normalize_match_text(term) in structured_text for term in terms)

        if q_has("过去4天", "超时", "还不发货", "不发货吗"):
            if k_has("已超时", "催促", "催仓库", "帮您催"):
                score += 70
            if k_has("48小时内发货", "48 小时内发货"):
                score -= 45

        if q_has("给我录一下风", "风是有多大", "风有多大", "风力有多大", "风力大吗"):
            if k_has("风力", "风速", "档位", "风力噪音"):
                score += 70
            if k_has("尺寸", "重量", "物理尺寸", "大小"):
                score -= 70

        if query_clean in {"多长", "长度多长", "有多长"}:
            if k_has("尺寸", "长度", "物理尺寸", "57*56*160", "160mm"):
                score += 85
            if k_has("续航", "小时", "多久", "使用时间"):
                score -= 75

        if q_has("电机", "无刷", "有刷", "静音电机"):
            if k_has("电机类型", "无刷电机", "无刷"):
                score += 85
            elif k_has("静音特性", "噪音", "风噪"):
                score -= 35

        if q_has("异味", "塑料味", "味道") and not q_has("噪音", "声音", "吵"):
            if k_has("异味", "塑料味", "味道"):
                score += 75
            if k_has("噪音大", "噪音与其他问题"):
                score -= 55

        if q_has("档位", "几档", "最高几档", "几档调节", "有几档", "档位多少"):
            if k_has("档位数量", "版本区别", "规格区别"):
                score += 85

        if q_has("长续航", "高配", "旗舰", "最大规格"):
            if k_has("版本名称", "版本区别", "不等于实际", "长续航版本使用时间"):
                score += 45

        if query_clean == "续航":
            if k_has("续航时间标准", "续航一般", "续航时间查询"):
                score += 65
            if k_has("续航短", "不符质疑", "虚电"):
                score -= 45

        if query_clean == "发票":
            if k_has("发票开具咨询", "支持开普通发票"):
                score += 65
            if k_has("之前说", "30日内", "登机"):
                score -= 45

        if q_has("我要退款", "直接退款", "帮我退款", "退钱", "我要补偿", "补偿一下", "能赔吗", "不想要了"):
            has_reason_signal = q_has(
                "原因", "坏", "不能用", "不转", "没反应", "打不开", "开不了",
                "充不了", "充不进", "风力", "噪音", "声音大", "续航", "异味",
                "发热", "破损", "裂", "松动", "发错", "少配件", "少了",
            )
            if not has_reason_signal:
                if k_has("用户直接请求退款未提供原因", "未提供原因", "是什么原因不想要"):
                    score += 160
                if k_has("售后补充-场景识别边界") and k_has("直接申请退款即可"):
                    score -= 200
                if k_has("用户询问补偿或退款的选择"):
                    score -= 35

        return score

    @classmethod
    def _qualifier_groups(cls) -> tuple[tuple[str, ...], ...]:
        groups = (
            ("最大档", "最高档", "最大档位", "最高档位", "旗舰"),
            ("最低档", "低档", "最小风"),
            ("长续航", "高配", "旗舰"),
            ("中续航", "中配", "标准版"),
            ("短续航", "低配", "基础版"),
        )
        return tuple(
            tuple(cls._normalize_match_text(term) for term in group)
            for group in groups
        )

    @classmethod
    def _detect_query_scenario(cls, text: str) -> str:
        clean = cls._normalize_match_text(text)
        if not clean:
            return ""

        def has_any(words: tuple[str, ...]) -> bool:
            return any(cls._normalize_match_text(word) in clean for word in words)

        scenarios = (
            ("charge_power", (
                "充电", "充电器", "充电头", "充电线", "数据线", "typec", "type-c", "接口",
                "充电口", "电量", "边充边用", "充不了电", "充不进电", "充不上电",
                "不能充电", "无法充电", "没法充电", "不充电", "充电没反应", "充电不亮",
            )),
            ("color_purchase", ("颜色", "白色", "黑色", "绿色", "紫色", "有货", "现货", "库存", "混色", "备注", "一黑一白")),
            ("product_usage", ("挂脖", "挂绳", "折叠", "桌面", "开关", "怎么用", "使用教程", "说明书")),
            ("cooling", ("制冷", "冰敷", "结冰", "制冰", "半导体", "凉感")),
            ("noise", ("静音", "噪音", "声音", "分贝", "吵")),
            ("shipping", ("快递", "发货", "物流", "到货", "发货地", "从哪发", "从哪里发")),
            ("aftersale", ("质保", "保修", "退货", "退款", "运费", "运费险", "质量问题", "坏了")),
            ("battery_endurance", ("续航", "几个小时", "多久", "毫安", "电池", "充满")),
            ("wind_power", ("风力", "风速", "档位", "几档", "最大档", "最高档")),
            ("size_weight", ("尺寸", "多大", "重量", "多重", "厘米")),
        )
        for name, words in scenarios:
            if has_any(words):
                return name
        return ""

    @classmethod
    def _detect_query_intent(cls, text: str) -> str:
        clean = cls._normalize_match_text(text)
        if not clean:
            return ""

        def has_any(words: tuple[str, ...]) -> bool:
            return any(cls._normalize_match_text(word) in clean for word in words)

        charger_words = ("充电器", "充电头")
        cable_words = ("充电线", "数据线", "线")
        gift_words = ("送", "赠", "带", "有", "里面", "包装", "配", "附")
        compat_words = ("能用", "可以用", "通用", "手机", "华为", "普通", "什么", "哪种")
        method_words = ("怎么充电", "如何充电", "什么接口", "充电口", "typec", "type-c", "接口")
        charge_abnormal_words = (
            "充不了电", "充不进电", "充不上电", "不能充电", "无法充电",
            "没法充电", "不充电", "充电没反应", "充电不亮", "充电异常",
        )
        color_words = ("颜色", "色", "白色", "黑色", "绿色", "紫色", "几种颜色", "什么颜色")
        stock_words = ("有货", "现货", "能拍", "拍下", "库存")
        mix_order_words = ("一黑一白", "混合颜色", "混色", "备注", "发一黑一白", "发两个颜色")
        hang_words = ("挂脖", "挂绳", "挂着")
        cool_words = ("制冷", "冰敷", "结冰", "制冰", "凉不凉", "半导体")
        noise_words = ("静音", "噪音", "声音", "分贝", "吵")
        shipping_time_words = (
            "什么时候发货", "多久发货", "几天到", "什么时候到", "多久到", "加急",
            "还不发货", "不发货", "没发货", "货发了没有", "催发货", "尽快发货",
            "快点发货",
        )
        shipping_express_words = ("什么快递", "发啥快递", "哪家快递", "快递")
        shipping_origin_words = ("发货地", "哪里发货", "从哪发货", "从哪里发货")
        warranty_words = ("质保", "保修", "坏了怎么办", "质量问题怎么办")
        return_shipping_words = ("退货包运费", "运费谁出", "运费险")
        return_policy_words = ("可以退货吗", "退货政策", "退款", "7天无理由")
        endurance_words = ("续航", "能用多久", "充满多久", "几个小时")
        capacity_words = ("多少毫安", "电池多大", "电池容量")
        speed_words = ("几档", "档位", "风速", "风力", "最大档", "最高档")
        size_words = ("尺寸", "多大", "多重", "重量", "几厘米")

        if has_any(charger_words) and has_any(gift_words):
            return "gift_charger"
        if has_any(cable_words) and has_any(gift_words):
            return "gift_cable"
        if has_any(charger_words) and has_any(compat_words):
            return "charger_compat"
        if has_any(cable_words) and has_any(compat_words):
            return "cable_compat"
        if has_any(charge_abnormal_words):
            return "charge_abnormal"
        if has_any(method_words):
            return "charge_method"
        if has_any(color_words) and has_any(mix_order_words):
            return "mixed_color_order"
        if has_any(color_words) and has_any(stock_words):
            return "color_stock"
        if has_any(color_words):
            return "color_query"
        if has_any(hang_words):
            return "hang_support"
        if has_any(cool_words):
            return "cooling"
        if has_any(noise_words):
            return "noise"
        if has_any(shipping_origin_words):
            return "shipping_origin"
        if has_any(shipping_express_words):
            return "shipping_express"
        if has_any(shipping_time_words):
            return "shipping_time"
        if has_any(warranty_words):
            return "warranty"
        if has_any(return_shipping_words):
            return "return_shipping"
        if has_any(return_policy_words):
            return "return_policy"
        if has_any(capacity_words):
            return "battery_capacity"
        if has_any(endurance_words):
            return "endurance"
        if has_any(speed_words):
            return "wind_speed"
        if has_any(size_words):
            return "size_weight"
        return ""

    @staticmethod
    def _normalize_scenario_name(name: str) -> str:
        mapping = {
            "charge_power": "充电用电",
            "color_purchase": "购买相关",
            "product_usage": "产品使用",
            "cooling": "制冷功能",
            "noise": "静音噪音",
            "shipping": "发货物流",
            "aftersale": "退换货售后",
            "battery_endurance": "续航电池",
            "wind_power": "风力风速",
            "size_weight": "尺寸重量",
        }
        return mapping.get(name or "", name or "")

    def _rank_meta_entries(
        self,
        entries: List[KnowledgeMetaEntry],
        query: str,
        limit: int,
        scene_key: str = "",
    ) -> List[KnowledgeMetaEntry]:
        scored = []
        for index, entry in enumerate(entries):
            scenario_label = self._normalize_scenario_name(entry.scenario)
            sub_intent_label = entry.sub_intent or ""
            match_score = self._structured_match_score(
                query=query,
                aliases=entry.aliases or "",
                answer=entry.answer or "",
                section=scenario_label,
                tags=f"{entry.tags or ''} {sub_intent_label}",
            )
            if match_score <= 0:
                continue
            score = match_score + min(int(entry.priority or 0), 20)
            scene_score = self._customer_scene_match_score(
                scene_key,
                scenario_label,
                sub_intent_label,
                entry.section_title or "",
                entry.tags or "",
            )
            if scene_score > 0:
                score += scene_score
            elif scene_score < 0:
                score += scene_score
            if scene_key:
                tags_text = entry.tags or ""
                scene_text = " ".join(
                    str(part or "")
                    for part in (scenario_label, entry.section_title, entry.sub_intent, tags_text)
                )
                if "scene_kb" in tags_text:
                    score += 35
                elif not self._primary_customer_scene(scene_text):
                    score -= 45
            if (
                scene_key
                and "售前同步" in (entry.tags or "")
                and "补充" in (entry.section_title or "")
                and not self._has_exact_alias_match(query, entry.aliases or "")
            ):
                score -= 12
            if score > 0:
                scored.append((score, getattr(entry, "id", 0) or 0, index, entry))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [entry for score, _, _, entry in scored[:limit] if score >= 8]

    @staticmethod
    def _search_terms(query: str) -> List[str]:
        text = str(query or "").strip()
        if not text:
            return []

        phrase_candidates = (
            "续航", "充满电", "充满", "能用多久", "用多久", "多长时间", "几个小时",
            "最大档", "最高档", "最低档", "低档", "高档", "电池容量", "多少毫安",
            "长续航", "中续航", "短续航", "高配", "中配", "基础版", "充电线", "数据线", "充电器", "充电头",
            "送充电器", "送充电头", "送充电线", "有充电器", "有充电头", "有充电线",
            "充不了电", "充不进电", "充不上电", "不能充电", "无法充电", "不充电",
            "手机充电器", "普通充电器", "Type-C", "typec", "type-c", "充电口", "什么接口",
            "静音", "噪音", "声音", "分贝", "制冷", "冰敷", "风力", "风速", "档位",
            "尺寸", "重量", "挂绳", "挂脖", "赠品", "颜色", "黑色", "白色", "绿色", "紫色",
            "有货", "现货", "什么快递", "发货地", "质保", "保修", "退货包运费",
        )
        raw_terms = []
        raw_terms.extend(word.strip() for word in jieba.cut_for_search(text))
        raw_terms.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text))
        raw_terms.extend(phrase for phrase in phrase_candidates if phrase.lower() in text.lower())
        if re.search(r"充(?:不|不了|不上|不进|不进去)|(?:不能|无法|没法)充电|不充电", text):
            raw_terms.extend(["充电", "充电异常"])

        stop_words = {
            "多久", "多少", "什么", "怎么", "可以", "有没有", "是不是",
            "这个", "那个", "一下", "大概", "请问", "亲", "需要", "帮我",
        }
        terms: List[str] = []
        for term in raw_terms:
            clean = term.strip()
            if len(clean) < 2 or clean in stop_words or clean in terms:
                continue
            terms.append(clean)
        return terms

    @staticmethod
    def _product_knowledge_blocks(content: str) -> List[str]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        blocks = []
        section = ""
        index = 0
        parameter_keywords = (
            "功率", "转速", "风速", "续航", "充电", "赠品", "尺寸",
            "重量", "档位", "电池", "容量", "快递", "发货",
        )

        while index < len(lines):
            line = lines[index]
            if line.startswith("##"):
                section = line
                index += 1
                continue

            if line.startswith("- 问法") or line.startswith("问法"):
                block_lines = [section] if section else []
                block_lines.append(line)
                next_index = index + 1
                if next_index < len(lines) and "要点" in lines[next_index]:
                    block_lines.append(lines[next_index])
                    index += 2
                else:
                    index += 1
                blocks.append("\n".join(block_lines))
                continue

            if line.startswith("- ") and any(keyword in line for keyword in parameter_keywords):
                block_lines = [section, line] if section else [line]
                aliases = KnowledgeService._parameter_aliases(line)
                if aliases:
                    block_lines.append(aliases)
                blocks.append("\n".join(block_lines))

            index += 1

        return blocks

    @staticmethod
    def _parameter_aliases(line: str) -> str:
        aliases = []
        if "功率" in line:
            aliases.append("问法：功率多少瓦/几瓦/多少W/功率多大")
        if "转速" in line:
            aliases.append("问法：转速多少/每分钟多少转/转速快吗")
        if "风速" in line:
            aliases.append("问法：风速多少/最大风速多少/多少米每秒")
        if "续航" in line or "电池" in line or "容量" in line:
            aliases.append("问法：续航多久/电池容量多大/能用多长时间/充满用多久")
        if "充电" in line or "Type-C" in line:
            aliases.append("问法：怎么充电/送充电线吗/可以边充边用吗")
        return "；".join(aliases)

    @staticmethod
    def _strip_embedding_aliases(block: str) -> str:
        lines = [
            line
            for line in block.splitlines()
            if not re.match(r"^-?\s*问法[:：]", line.strip())
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _merge_text_blocks(primary: List[str], fallback: List[str], limit: int) -> str:
        merged = []
        seen = set()
        for block in [*primary, *fallback]:
            clean_block = block.strip()
            if not clean_block or clean_block in seen:
                continue
            seen.add(clean_block)
            merged.append(clean_block)
            if len(merged) >= limit:
                break
        return "\n\n".join(merged)

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 420, overlap_chars: int = 80) -> List[str]:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        chunks = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                start = 0
                step = max_chars - overlap_chars
                while start < len(paragraph):
                    chunks.append(paragraph[start:start + max_chars])
                    start += step
                continue

            candidate = f"{current}\n{paragraph}".strip() if current else paragraph
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _keyword_product_content(content: str, query: str, limit: int) -> str:
        words = KnowledgeService._search_terms(query)
        if not words:
            return ""
        stop_words = {
            "多久", "多少", "什么", "怎么", "可以", "有没有", "是不是",
            "这个", "那个", "一下", "大概", "能用", "请问",
        }
        match_words = [word for word in words if word not in stop_words] or words
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        matched_blocks = []
        seen = set()
        for index, line in enumerate(lines):
            if any(word in line for word in match_words):
                block_lines = []
                if index > 0 and lines[index - 1].startswith("###"):
                    block_lines.append(lines[index - 1])
                block_lines.append(line)
                for offset in range(1, 3):
                    next_index = index + offset
                    if next_index < len(lines):
                        next_line = lines[next_index]
                        if next_line.startswith("要点") or next_line.startswith("- 要点") or "要点：" in next_line:
                            block_lines.append(next_line)
                            break
                block = "\n".join(block_lines)
                if block not in seen:
                    seen.add(block)
                    matched_blocks.append(block)
            if len(matched_blocks) >= limit:
                break
        return "\n\n".join(matched_blocks)

    @staticmethod
    def _merge_ranked_results(primary: List[Any], fallback: List[Any], limit: int) -> List[Any]:
        merged = []
        seen = set()
        for item in [*primary, *fallback]:
            item_id = getattr(item, "id", id(item))
            if item_id in seen:
                continue
            seen.add(item_id)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _infer_product_family(goods_name: str) -> str:
        return ""

    @classmethod
    def normalize_customer_scene(cls, scene: Optional[str]) -> str:
        """归一化客服大场景，返回 presale/insale/aftersale。"""
        clean = cls._normalize_match_text(scene or "")
        if not clean:
            return ""
        direct = {
            "售前": "presale",
            "presale": "presale",
            "pre_sale": "presale",
            "pre-sale": "presale",
            "售中": "insale",
            "insale": "insale",
            "in_sale": "insale",
            "in-sale": "insale",
            "售后": "aftersale",
            "aftersale": "aftersale",
            "after_sale": "aftersale",
            "after-sale": "aftersale",
        }
        if clean in {cls._normalize_match_text(key) for key in direct}:
            for key, value in direct.items():
                if clean == cls._normalize_match_text(key):
                    return value
        for scene_key, aliases in CUSTOMER_SCENE_ALIASES.items():
            if any(cls._normalize_match_text(alias) in clean for alias in aliases):
                return scene_key
        return ""

    @classmethod
    def customer_scene_label(cls, scene: Optional[str]) -> str:
        scene_key = cls.normalize_customer_scene(scene) or str(scene or "")
        return CUSTOMER_SCENE_LABELS.get(scene_key, scene_key or "售前")

    @classmethod
    def detect_customer_scene(cls, text: str, default: str = "presale") -> str:
        """从预处理后的订单/会话文本里识别客服大场景。"""
        clean = cls._normalize_match_text(text)
        if not clean:
            return cls.normalize_customer_scene(default)

        def has_any(words: tuple[str, ...]) -> bool:
            return any(cls._normalize_match_text(word) in clean for word in words)

        if has_any(("当前业务场景：售后倾向", "当前业务场景售后倾向", "当前订单状态：已签收", "当前订单状态已签收", "已签收")):
            return "aftersale"
        if has_any((
            "当前业务场景：售中-待发货", "当前业务场景：售中-物流中",
            "当前业务场景售中待发货", "当前业务场景售中物流中",
            "当前订单状态：待发货", "当前订单状态：已发货待收货",
            "当前订单状态待发货", "当前订单状态已发货待收货",
        )):
            return "insale"
        if has_any(("售前咨询", "购买前", "下单前", "拍前", "买前")):
            return "presale"

        direct_aftersale_words = (
            "我要退款", "我要退货", "我要退货退款", "退货退款", "退款退货",
            "退款", "退款不退货", "申请退货", "申请退款", "申请退货退款", "申请退货退款吧",
            "给我退款", "给我退钱", "我希望你退款", "想退款", "退钱", "仅退款",
            "退货", "退的话", "如果退", "能退吗", "可以退吗", "还能退吗",
            "不想要了", "不要了", "退了吧", "周末可以退", "周末再退",
            "现在申请", "退不了", "没法退", "不能退", "补偿", "赔偿", "退差价",
            "给我退", "要求退款", "运费险多少钱", "运费多少钱", "包运费险吗",
            "包运费吗", "退货包运费", "运费谁出", "退货运费",
        )
        if has_any(direct_aftersale_words):
            return "aftersale"

        received_words = ("收到", "到货", "签收", "刚拿到", "用了", "使用了", "买的")
        problem_words = (
            "打不开", "开不了", "不转", "没反应", "不能用", "用不了",
            "不出风", "没风", "不能吹", "不吹风", "突然不能吹",
            "充不了", "充不了电", "充不进", "充不进电", "不充电", "无法充电", "坏了", "坏的",
            "开关没反应", "开关没有反应", "开关还是没有反应", "一拔充电器没有反应",
            "拔充电器没有反应", "重新充电归零", "又归零", "不保电",
            "声音大", "声音太大", "声音很大", "噪音大", "噪音太大", "噪音很大",
            "风力小", "风力太小", "风太小", "风小", "不凉", "不够凉",
            "续航短", "续航太短", "续航不行", "异味", "有异味", "有味道",
            "发热", "很烫", "破损", "破了", "裂开", "裂了", "开裂", "松动",
            "少配件", "少件", "少东西", "配件少", "发错",
        )
        presale_quality_questions = (
            "有噪音吗", "声音大吗", "声音大不大", "噪音大吗", "噪音大不大",
            "静音吗", "是不是真的静音", "会不会很吵", "吵不吵",
            "风力大吗", "风力大不大", "风大吗", "风大不大", "凉快吗",
        )
        quality_complaint_words = (
            "声音大跟", "声音大了", "声音比较大", "声音大的", "声音很吵", "声音太吵",
            "声音怎么这么大", "怎么这么大声音", "怎么声音这么大",
            "声特别大", "声音特别大", "声音特别响", "声音好大", "噪音好大",
            "噪音这么大", "噪音怎么这么大", "怎么这么大噪音", "噪音太大", "噪音很大", "有噪音啊", "有噪音了",
            "声音这么大", "声音太大", "声音很大", "声音还很大", "声音挺大",
            "电机声音大", "电机声音也挺大", "电机比较响", "电机大声", "不是静音", "不静音", "还静音呢",
            "吵聋", "吵死", "太吵", "很吵", "耳朵吵",
            "噪音有点大", "噪音比较大", "噪音大了", "声音有点大",
            "为什么声音", "为什么不是静音", "风力还是太小", "风力太小", "风力很小",
            "风力不好", "风太小", "风很小", "没什么风", "也没什么风", "没有风",
            "根本感觉不到风", "感觉不到风", "没有感受到一点风", "一点风",
            "一点点这个风", "不够凉", "不凉快", "一点都不凉", "一点都不凉快",
            "贴近脸", "跟没吹一样", "根本就不能用", "一点都不能用",
            "都用了不好使", "都用了，不好使", "用了不好使", "买回来就不好使",
            "用不到1小时", "不到1小时", "一点点这个风", "一会儿就没电", "一下就没电",
            "很快就没电", "才用", "40分钟", "四十分钟", "掉电", "掉了十个点",
            "直接掉了十个点", "续航太差", "续航不行", "续航短", "续航太短", "不保电",
            "塑料味", "臭味", "有味道", "味道很大", "烧焦味", "发烫", "很烫",
        )
        if has_any(quality_complaint_words) and not has_any(presale_quality_questions):
            return "aftersale"
        if has_any(("都用了不好使", "都用了，不好使", "用了不好使", "买回来就不好使", "刚拿到就不好使")):
            return "aftersale"
        if has_any(received_words) and has_any(problem_words):
            return "aftersale"

        direct_problem_words = (
            "打不开", "开不了", "不转", "没反应", "不能用", "用不了", "不好使",
            "不出风", "没风", "不能吹", "不吹风", "突然不能吹", "吹不了风",
            "充不了电", "充不进电", "充不去电", "不充电", "无法充电", "坏了", "坏的",
            "开关没反应", "开关没有反应", "开关还是没有反应", "一拔充电器没有反应",
            "拔充电器没有反应", "重新充电归零", "又归零", "不保电",
            "破损", "破了", "裂开", "裂了", "开裂", "碎的", "断了", "少配件", "少件", "发错",
        )
        if has_any(direct_problem_words):
            return "aftersale"

        problem_followup_words = ("怎么办", "咋办", "怎么处理", "怎么解决", "处理一下", "给处理", "补偿", "赔", "退")
        if has_any(problem_words) and has_any(problem_followup_words):
            return "aftersale"

        order_words = ("我的", "订单", "下单", "拍了", "买了", "还没", "怎么还", "催", "加急")
        fulfillment_words = ("发货", "物流", "快递", "地址", "拦截", "截回", "到哪")
        direct_insale_action_words = (
            "地址填错", "地址错", "改地址", "换地址", "拦截", "截回", "拒收", "改派", "取件码",
            "不是说后天", "两个后天", "还没送达", "还没到", "没收到货", "没到货",
            "什么时候送达", "什么时候到", "到哪了", "物流不动", "一直没动",
        )
        if has_any(direct_insale_action_words):
            return "insale"
        if has_any(order_words) and has_any(fulfillment_words):
            return "insale"

        return cls.normalize_customer_scene(default)

    @classmethod
    def _customer_scene_match_score(cls, scene: Optional[str], *texts: str) -> int:
        scene_key = cls.normalize_customer_scene(scene)
        if not scene_key:
            return 0
        raw_text = " ".join(str(text or "") for text in texts)
        primary_scene = cls._primary_customer_scene(raw_text)
        if primary_scene:
            return 48 if primary_scene == scene_key else -48

        clean = cls._normalize_match_text(raw_text)
        if not clean:
            return 0

        hit_keys = []
        for key, aliases in CUSTOMER_SCENE_ALIASES.items():
            if any(cls._normalize_match_text(alias) in clean for alias in aliases):
                hit_keys.append(key)
        if scene_key not in hit_keys:
            return -12 if hit_keys else 0
        # 只有一个明确场景时强加权；同时出现多个场景说明是兼容规则，不强排除。
        return 36 if len(hit_keys) == 1 else 8

    @classmethod
    def _primary_customer_scene(cls, text: str) -> str:
        """识别知识条目的主场景，避免“售前同步”把跨场景副本当成原生售前。"""
        clean_text = str(text or "")
        if not clean_text.strip():
            return ""

        for token in re.split(r"[,，/|;；\s]+", clean_text):
            clean_token = cls._normalize_match_text(token)
            if clean_token in ("售前", "presale"):
                return "presale"
            if clean_token in ("售中", "insale"):
                return "insale"
            if clean_token in ("售后", "aftersale"):
                return "aftersale"

        compact = cls._normalize_match_text(clean_text)
        if compact.startswith("售前补充"):
            return "presale"
        if compact.startswith("售中补充"):
            return "insale"
        if compact.startswith("售后补充"):
            return "aftersale"
        return ""

    def _filter_customer_service_by_scene(
        self,
        cs_list: List[CustomerServiceKnowledge],
        scene: Optional[str],
        fallback_to_all: bool = True,
    ) -> List[CustomerServiceKnowledge]:
        scene_key = self.normalize_customer_scene(scene)
        if not scene_key:
            return cs_list
        scored = []
        for index, cs in enumerate(cs_list):
            score = self._customer_scene_match_score(scene_key, cs.title or "", cs.tags or "")
            if score >= 0:
                scored.append((score, index, cs))
        if not scored:
            return cs_list if fallback_to_all else []
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [cs for _, _, cs in scored]

    def _filter_meta_entries_by_customer_scene(
        self,
        entries: List[KnowledgeMetaEntry],
        scene: Optional[str],
        fallback_to_all: bool = True,
    ) -> List[KnowledgeMetaEntry]:
        scene_key = self.normalize_customer_scene(scene)
        if not scene_key:
            return entries
        scored = []
        for index, entry in enumerate(entries):
            score = self._customer_scene_match_score(
                scene_key,
                self._normalize_scenario_name(entry.scenario),
                entry.sub_intent or "",
                entry.section_title or "",
                entry.tags or "",
            )
            if score >= 0:
                scored.append((score, index, entry))
        if not scored:
            return entries if fallback_to_all else []
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [entry for _, _, entry in scored]

    def get_full_scene_customer_service_knowledge(
        self,
        shop_id: int,
        goods_id: int,
        scene: Optional[str],
    ) -> str:
        """按商品ID和当前场景读取完整客服知识，用于RAG未命中后的兜底注入。"""
        scene_key = self.normalize_customer_scene(scene) or "presale"
        db_shop_id = self._resolve_shop_id(shop_id)

        with self.get_session() as session:
            meta_entries = list(session.scalars(
                select(KnowledgeMetaEntry).where(
                    and_(
                        KnowledgeMetaEntry.shop_id == db_shop_id,
                        KnowledgeMetaEntry.source_type == "customer_service",
                        KnowledgeMetaEntry.goods_id == goods_id,
                        KnowledgeMetaEntry.enabled == True,
                    )
                )
            ))
            scene_entries = self._filter_meta_entries_by_customer_scene(
                meta_entries,
                scene_key,
                fallback_to_all=False,
            )
            if not scene_entries:
                return ""

            ordered_source_ids = []
            seen_source_ids = set()
            for entry in sorted(scene_entries, key=lambda item: (-(item.priority or 0), item.id or 0)):
                if entry.source_id in seen_source_ids:
                    continue
                seen_source_ids.add(entry.source_id)
                ordered_source_ids.append(entry.source_id)

            if not ordered_source_ids:
                return ""

            cs_rows = list(session.scalars(
                select(CustomerServiceKnowledge).where(
                    and_(
                        CustomerServiceKnowledge.shop_id == db_shop_id,
                        CustomerServiceKnowledge.enabled == True,
                        CustomerServiceKnowledge.id.in_(ordered_source_ids),
                    )
                )
            ))
            cs_by_id = {item.id: item for item in cs_rows}
            ordered_cs = [cs_by_id[source_id] for source_id in ordered_source_ids if source_id in cs_by_id]
            if not ordered_cs:
                return ""

            output_parts = ["【客服知识】"]
            for index, cs in enumerate(ordered_cs, 1):
                title = (cs.title or "").split("/")[0].strip() or "命中客服知识"
                output_parts.append(f"{index}. {title}\n  {cs.content or ''}")
            return "\n\n".join(output_parts).strip()

    @staticmethod
    def _keyword_customer_service_entries(
        cs_list: List[CustomerServiceKnowledge],
        words: List[str],
        limit: int,
    ) -> List[CustomerServiceKnowledge]:
        if not words:
            return []
        matched = []
        lowered_words = [word.lower() for word in words if word]
        for index, cs in enumerate(cs_list):
            text = "\n".join([cs.title or "", cs.tags or "", cs.content or ""]).lower()
            if all(word in text for word in lowered_words):
                created_at = getattr(cs, "created_at", None) or datetime.min
                matched.append((created_at, index, cs))
        matched.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [cs for _, _, cs in matched[:limit]]

    def _search_customer_service_candidates(
        self,
        all_cs: List[CustomerServiceKnowledge],
        candidate_cs: List[CustomerServiceKnowledge],
        candidate_meta_entries: List[KnowledgeMetaEntry],
        query: str,
        words: List[str],
        db_shop_id: int,
        limit: int,
        scene: Optional[str] = None,
    ) -> List[CustomerServiceKnowledge]:
        scene_key = self.normalize_customer_scene(scene)
        ranked_meta_cs = self._rank_meta_entries(candidate_meta_entries, query, limit, scene_key=scene_key)
        if ranked_meta_cs:
            meta_source_ids = [entry.source_id for entry in ranked_meta_cs]
            structured_cs = []
            for source_id in meta_source_ids:
                match = next((item for item in all_cs if item.id == source_id), None)
                if match:
                    structured_cs.append(match)
        else:
            structured_cs = self._rank_customer_service_entries(
                candidate_cs,
                query,
                limit,
                scene_key=scene_key,
            )

        if structured_cs:
            return self._filter_relevant_customer_service_entries(
                structured_cs[:1],
                query,
                1,
                scene_key=scene_key,
            )

        vector_cs = self.vector_retriever.rank(
            namespace=f"customer_service_knowledge_{scene_key or 'all'}",
            shop_id=db_shop_id,
            query=query,
            items=self._customer_service_vector_items(candidate_cs),
            limit=limit,
        )
        keyword_cs = self._keyword_customer_service_entries(candidate_cs, words, limit)

        merged_cs = self._merge_ranked_results(vector_cs, keyword_cs, limit)
        return self._filter_relevant_customer_service_entries(
            merged_cs,
            query,
            limit,
            scene_key=scene_key,
        )

    def _rank_customer_service_entries(
        self,
        cs_list: List[CustomerServiceKnowledge],
        query: str,
        limit: int,
        scene_key: str = "",
    ) -> List[CustomerServiceKnowledge]:
        scored = []
        for index, cs in enumerate(cs_list):
            tags = cs.tags or ""
            match_score = self._structured_match_score(
                query=query,
                aliases=cs.title or "",
                answer=cs.content or "",
                tags=tags,
            )
            if match_score <= 0:
                continue
            score = match_score
            if "faq_split" in tags:
                score += 4
            scene_score = self._customer_scene_match_score(scene_key, cs.title or "", tags)
            if scene_score > 0:
                score += scene_score
            elif scene_score < 0:
                score += scene_score
            if scene_key:
                tags_text = cs.tags or ""
                scene_text = " ".join(str(part or "") for part in (cs.title, tags_text))
                if "scene_kb" in tags_text:
                    score += 25
                elif not self._primary_customer_scene(scene_text):
                    score -= 35
            if score > 0:
                scored.append((score, getattr(cs, "id", 0) or 0, index, cs))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [cs for score, _, _, cs in scored[:limit] if score >= 8]

    def _filter_relevant_customer_service_entries(
        self,
        cs_list: List[CustomerServiceKnowledge],
        query: str,
        limit: int,
        scene_key: str = "",
    ) -> List[CustomerServiceKnowledge]:
        if not cs_list or not str(query or "").strip():
            return cs_list[:limit]

        scored = []
        for index, cs in enumerate(cs_list):
            match_score = self._structured_match_score(
                query=query,
                aliases=cs.title or "",
                answer=cs.content or "",
                tags=cs.tags or "",
            )
            if match_score <= 0:
                continue
            score = match_score
            scene_score = self._customer_scene_match_score(scene_key, cs.title or "", cs.tags or "")
            if scene_score > 0:
                score += scene_score
            elif scene_score < 0:
                score += scene_score
            if scene_key:
                tags_text = cs.tags or ""
                scene_text = " ".join(str(part or "") for part in (cs.title, tags_text))
                if "scene_kb" in tags_text:
                    score += 25
                elif not self._primary_customer_scene(scene_text):
                    score -= 35
            if score >= 18:
                scored.append((score, getattr(cs, "id", 0) or 0, index, cs))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [cs for _, _, _, cs in scored[:limit]]

    def search_knowledge(
        self,
        shop_id: int,
        query: Optional[str] = None,
        goods_id: Optional[int] = None,
        limit: int = 10,
        search_scope: str = "all",
        scene: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {
            "product_knowledge": [],
            "customer_service_knowledge": [],
        }
        db_shop_id = self._resolve_shop_id(shop_id)

        with self.get_session() as session:
            product: Optional[ProductKnowledge] = None
            if goods_id is not None:
                product_stmt = select(ProductKnowledge).where(
                    and_(
                        ProductKnowledge.shop_id == db_shop_id,
                        ProductKnowledge.goods_id == goods_id,
                    )
                )
                product = session.scalar(product_stmt)

            if goods_id is not None and search_scope in ("all", "product") and product:
                result["product_knowledge"] = [product]
                force_full_content = False
                meta_entries = list(session.scalars(
                    select(KnowledgeMetaEntry).where(
                        and_(
                            KnowledgeMetaEntry.shop_id == db_shop_id,
                            KnowledgeMetaEntry.source_type == "product",
                            KnowledgeMetaEntry.goods_id == goods_id,
                            KnowledgeMetaEntry.enabled == True,
                        )
                    )
                ))
                ranked_meta = self._rank_meta_entries(meta_entries, query or "", limit=1)
                if ranked_meta:
                    meta = ranked_meta[0]
                    matched_content = (
                        f"### {self._normalize_scenario_name(meta.scenario)}\n"
                        f"{meta.answer}"
                    )
                    if len(matched_content.strip()) < MIN_PRODUCT_HIT_CHARS:
                        fallback_content = self._rank_product_content(product, query, limit=5)
                        if fallback_content and len(fallback_content.strip()) > len(matched_content.strip()):
                            matched_content = fallback_content
                            force_full_content = False
                        else:
                            force_full_content = True
                else:
                    matched_content = self._rank_product_content(product, query, limit=5)
                    if matched_content and len(matched_content.strip()) < MIN_PRODUCT_HIT_CHARS:
                        force_full_content = True
                if matched_content:
                    result["product_knowledge_hits"] = {
                        product.id: matched_content,
                    }
                    result["product_force_full_content"] = {
                        product.id: force_full_content,
                    }
                else:
                    result["product_force_full_content"] = {
                        product.id: True,
                    }

            if goods_id is not None and search_scope == "product":
                return result

            if query and query.strip():
                words = [word.strip() for word in jieba.cut_for_search(query.strip()) if len(word.strip()) >= 2]

                if search_scope in ("all", "product"):
                    if goods_id is None or not product:
                        all_products_stmt = select(ProductKnowledge).where(ProductKnowledge.shop_id == db_shop_id)
                        all_products = list(session.scalars(all_products_stmt))
                        vector_products = self.vector_retriever.rank(
                            namespace="product_knowledge",
                            shop_id=db_shop_id,
                            query=query,
                            items=self._product_vector_items(all_products),
                            limit=limit,
                        )
                        keyword_products = []
                        if words:
                            product_conditions = [ProductKnowledge.shop_id == db_shop_id]
                            for word in words:
                                product_conditions.append(
                                    or_(
                                        ProductKnowledge.goods_name.contains(word),
                                        ProductKnowledge.extracted_content.contains(word),
                                    )
                                )
                            stmt_p = select(ProductKnowledge).where(and_(*product_conditions))\
                                .order_by(ProductKnowledge.created_at.desc())\
                                .limit(limit)
                            keyword_products = list(session.scalars(stmt_p))
                        result["product_knowledge"] = self._merge_ranked_results(
                            vector_products,
                            keyword_products,
                            limit,
                        )

                if search_scope in ("all", "customer_service"):
                    scene_key = self.normalize_customer_scene(scene) or self.detect_customer_scene(query, default="presale")
                    result["customer_service_scene"] = scene_key
                    customer_scope = "shop"
                    allow_shop_fallback = True
                    all_cs: List[CustomerServiceKnowledge] = []
                    meta_cs_entries: List[KnowledgeMetaEntry] = []
                    candidate_cs: List[CustomerServiceKnowledge] = []
                    candidate_meta_entries: List[KnowledgeMetaEntry] = []

                    if goods_id is not None and product:
                        product_family = self._infer_product_family(
                            " ".join(
                                part for part in [
                                    product.goods_name or "",
                                    product.specifications or "",
                                    product.extracted_content or "",
                                ]
                                if part
                            )
                        )
                        exact_meta_entries = list(session.scalars(
                            select(KnowledgeMetaEntry).where(
                                and_(
                                    KnowledgeMetaEntry.shop_id == db_shop_id,
                                    KnowledgeMetaEntry.source_type == "customer_service",
                                    KnowledgeMetaEntry.goods_id == goods_id,
                                    KnowledgeMetaEntry.enabled == True,
                                )
                            )
                        ))
                        if exact_meta_entries:
                            candidate_meta_entries = self._filter_meta_entries_by_customer_scene(
                                exact_meta_entries,
                                scene_key,
                                fallback_to_all=False,
                            )
                            source_ids = sorted({entry.source_id for entry in candidate_meta_entries})
                            if source_ids:
                                cs_stmt = select(CustomerServiceKnowledge).where(
                                    and_(
                                        CustomerServiceKnowledge.shop_id == db_shop_id,
                                        CustomerServiceKnowledge.enabled == True,
                                        CustomerServiceKnowledge.id.in_(source_ids),
                                    )
                                )
                                candidate_cs = list(session.scalars(cs_stmt))
                            customer_scope = f"goods_id:{goods_id}"
                            allow_shop_fallback = False
                        elif product_family:
                            family_meta_entries = list(session.scalars(
                                select(KnowledgeMetaEntry).where(
                                    and_(
                                        KnowledgeMetaEntry.shop_id == db_shop_id,
                                        KnowledgeMetaEntry.source_type == "customer_service",
                                        KnowledgeMetaEntry.product_family == product_family,
                                        KnowledgeMetaEntry.goods_id.is_(None),
                                        KnowledgeMetaEntry.enabled == True,
                                    )
                                )
                            ))
                            if family_meta_entries:
                                candidate_meta_entries = self._filter_meta_entries_by_customer_scene(
                                    family_meta_entries,
                                    scene_key,
                                    fallback_to_all=False,
                                )
                                source_ids = sorted({entry.source_id for entry in candidate_meta_entries})
                                if source_ids:
                                    cs_stmt = select(CustomerServiceKnowledge).where(
                                        and_(
                                            CustomerServiceKnowledge.shop_id == db_shop_id,
                                            CustomerServiceKnowledge.enabled == True,
                                            CustomerServiceKnowledge.id.in_(source_ids),
                                        )
                                    )
                                    candidate_cs = list(session.scalars(cs_stmt))
                                customer_scope = f"product_family:{product_family}"
                                allow_shop_fallback = False

                    if not candidate_meta_entries and not candidate_cs and allow_shop_fallback:
                        all_cs_stmt = select(CustomerServiceKnowledge).where(
                            and_(
                                CustomerServiceKnowledge.shop_id == db_shop_id,
                                CustomerServiceKnowledge.enabled == True,
                                or_(
                                    CustomerServiceKnowledge.tags.is_(None),
                                    ~CustomerServiceKnowledge.tags.contains("goods_id:"),
                                ),
                            )
                        )
                        all_cs = list(session.scalars(all_cs_stmt))
                        meta_cs_stmt = select(KnowledgeMetaEntry).where(
                            and_(
                                KnowledgeMetaEntry.shop_id == db_shop_id,
                                KnowledgeMetaEntry.source_type == "customer_service",
                                KnowledgeMetaEntry.goods_id.is_(None),
                                KnowledgeMetaEntry.enabled == True,
                            )
                        )
                        meta_cs_entries = list(session.scalars(meta_cs_stmt))
                        candidate_cs = self._filter_customer_service_by_scene(
                            all_cs,
                            scene_key,
                            fallback_to_all=True,
                        )
                        candidate_meta_entries = self._filter_meta_entries_by_customer_scene(
                            meta_cs_entries,
                            scene_key,
                            fallback_to_all=True,
                        )
                        customer_scope = "shop"
                        allow_shop_fallback = True

                    result["customer_service_scope"] = customer_scope
                    result["customer_service_knowledge"] = self._search_customer_service_candidates(
                        all_cs=candidate_cs or all_cs,
                        candidate_cs=candidate_cs,
                        candidate_meta_entries=candidate_meta_entries,
                        query=query,
                        words=words,
                        db_shop_id=db_shop_id,
                        limit=limit,
                        scene=scene_key,
                    )
                    if (
                        not result["customer_service_knowledge"]
                        and allow_shop_fallback
                        and scene_key
                        and (len(candidate_cs) < len(all_cs) or len(candidate_meta_entries) < len(meta_cs_entries))
                    ):
                        result["customer_service_knowledge"] = self._search_customer_service_candidates(
                            all_cs=all_cs,
                            candidate_cs=all_cs,
                            candidate_meta_entries=meta_cs_entries,
                            query=query,
                            words=words,
                            db_shop_id=db_shop_id,
                            limit=limit,
                            scene=None,
                        )
                        if result["customer_service_knowledge"]:
                            logger.info(
                                f"场景客服知识未命中，已回退全量检索: shop_id={db_shop_id}, "
                                f"scene={scene_key}, query={query!r}"
                            )
                return result

            if search_scope in ("all", "product"):
                stmt_p = select(ProductKnowledge).where(ProductKnowledge.shop_id == db_shop_id)\
                    .order_by(ProductKnowledge.created_at.desc())\
                    .limit(limit)
                result["product_knowledge"] = list(session.scalars(stmt_p))

            if search_scope in ("all", "customer_service"):
                stmt_cs = select(CustomerServiceKnowledge).where(
                    and_(
                        CustomerServiceKnowledge.shop_id == db_shop_id,
                        CustomerServiceKnowledge.enabled == True,
                    )
                ).order_by(CustomerServiceKnowledge.created_at.desc())\
                    .limit(limit)
                result["customer_service_knowledge"] = list(session.scalars(stmt_cs))

        return result

    def format_search_result(
        self,
        result: Dict[str, Any],
    ) -> str:
        """
        将检索结果格式化为Agent可读的字符串

        Args:
            result: search_knowledge 返回的结果

        Returns:
            格式化后的字符串
        """
        output_parts = []

        products = result.get("product_knowledge", [])
        product_hits = result.get("product_knowledge_hits", {})
        product_force_full_content = result.get("product_force_full_content", {})
        if products:
            output_parts.append("【产品知识】")
            for i, p in enumerate(products, 1):
                info = []
                info.append(f"{i}. {p.goods_name} (ID: {p.goods_id})")
                if p.price:
                    info.append(f"  价格: {p.price}")
                matched_content = product_hits.get(p.id)
                force_full_content = bool(product_force_full_content.get(p.id))
                if matched_content and not force_full_content:
                    info.append(f"  【与客户问题最相关的商品知识】\n  {matched_content}")
                elif p.extracted_content:
                    # 截断避免太长
                    content = p.extracted_content
                    max_content_length = 3200 if force_full_content else 1800
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    info.append(f"  {content}")
                output_parts.append("\n".join(info))
                output_parts.append("")

        cs_list = result.get("customer_service_knowledge", [])
        if cs_list:
            output_parts.append("【客服知识】")
            for i, cs in enumerate(cs_list, 1):
                info = []
                title = (cs.title or "").split("/")[0].strip() or "命中客服知识"
                info.append(f"{i}. {title}")
                content = cs.content
                if len(content) > 800:
                    content = content[:800] + "..."
                info.append(f"  {content}")
                output_parts.append("\n".join(info))
                output_parts.append("")

        if not output_parts:
            return "未找到相关知识。"

        return "\n".join(output_parts).strip()

    def get_all_shops(self) -> List[Shop]:
        """获取所有店铺列表（用于UI选择器）"""
        with self.get_session() as session:
            stmt = select(Shop).order_by(Shop.shop_name.asc())
            return list(session.scalars(stmt))

    # ========== 新场景知识检索 ==========

    _SCENE_MODEL_MAP = {
        "presale": PresaleKnowledge,
        "insale": InsaleKnowledge,
        "aftersale": AftersaleKnowledge,
    }

    def search_scene_knowledge(
        self,
        scene: str,
        shop_id: int,
        goods_id: Optional[int] = None,
        query: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        按场景检索知识（新表）。

        Args:
            scene: presale / insale / aftersale
            shop_id: 店铺 ID（原始值，会自动解析为 DB 内部 ID）
            goods_id: 商品 ID，可选
            query: 客户问题，可选
            limit: 返回条数，默认 3

        Returns:
            结果列表，每条包含 id/scene/goods_id/sub_intent/aliases/answer/
            section_title/tags/score/match_type/source_type/source_id/source_meta_id
        """
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model:
            logger.warning(f"search_scene_knowledge: 未知场景 '{scene}'，允许值: presale/insale/aftersale")
            return []

        db_shop_id = self._resolve_shop_id(shop_id)
        results: List[Dict[str, Any]] = []

        with self.get_session() as session:
            # ── 第一步：查商品专属知识 ──
            specific_entries: List = []
            if goods_id is not None:
                stmt = select(model).where(and_(
                    model.shop_id == db_shop_id,
                    model.goods_id == goods_id,
                    model.enabled == True,
                ))
                specific_entries = list(session.scalars(stmt))

            # ── 第二步：查店铺通用知识 ──
            generic_stmt = select(model).where(and_(
                model.shop_id == db_shop_id,
                model.goods_id.is_(None),
                model.enabled == True,
            ))
            generic_entries = list(session.scalars(generic_stmt))

            # ── 合并：专属优先 ──
            all_entries = specific_entries + generic_entries
            if not all_entries:
                return []

            # ── 排序 ──
            ranked = self._rank_scene_entries(
                all_entries,
                self._knowledge_match_query(query),
                scene_key,
                goods_id,
            )

            # ── 截取 top N ──
            for entry, score, match_type in ranked[:limit]:
                results.append({
                    "id": entry.id,
                    "scene": scene_key,
                    "goods_id": entry.goods_id,
                    "sub_intent": entry.sub_intent or "",
                    "aliases": entry.aliases or "",
                    "answer": entry.answer or "",
                    "section_title": entry.section_title or "",
                    "tags": entry.tags or "",
                    "score": score,
                    "match_type": match_type,
                    "source_type": entry.source_type or "",
                    "source_id": entry.source_id,
                    "source_meta_id": entry.source_meta_id,
                })

        return results

    def list_scene_knowledge_by_goods(
        self,
        scene: str,
        shop_id: int,
        goods_id: int,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List editable scene knowledge rows for one product."""
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model:
            return []

        db_shop_id = self._resolve_shop_id(shop_id)
        with self.get_session() as session:
            rows = list(session.scalars(
                select(model).where(and_(
                    model.shop_id == db_shop_id,
                    model.goods_id == goods_id,
                )).order_by(
                    model.priority.desc(),
                    model.section_title.asc(),
                    model.id.asc(),
                ).limit(limit)
            ))

        return [
            {
                "id": row.id,
                "scene": scene_key,
                "goods_id": row.goods_id,
                "sub_intent": row.sub_intent or "",
                "aliases": row.aliases or "",
                "answer": row.answer or "",
                "section_title": row.section_title or "",
                "tags": row.tags or "",
                "priority": row.priority or 0,
                "enabled": bool(row.enabled),
                "source_type": row.source_type or "",
                "source_id": row.source_id,
                "source_meta_id": row.source_meta_id,
            }
            for row in rows
        ]

    def update_scene_knowledge(
        self,
        scene: str,
        entry_id: int,
        aliases: str,
        answer: str,
        sub_intent: Optional[str] = None,
        section_title: Optional[str] = None,
        priority: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """Update one scene knowledge row from the management UI."""
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model:
            return False

        with self.get_session() as session:
            row = session.get(model, entry_id)
            if not row:
                return False
            row.aliases = aliases
            row.answer = answer
            row.sub_intent = sub_intent or ""
            row.section_title = section_title or ""
            if priority is not None:
                row.priority = int(priority)
            if enabled is not None:
                row.enabled = bool(enabled)
            row.updated_at = datetime.now()
            session.commit()
            return True

    def _rank_scene_entries(
        self,
        entries: List,
        query: Optional[str],
        scene_key: str,
        goods_id: Optional[int],
    ) -> List[tuple]:
        """
        对场景知识条目排序。

        商品专属作为第一排序键：所有专属条目排在通用条目之前，
        同组内按 score 降序。

        返回: [(entry, score, match_type), ...]
        """
        # 拆分专属 vs 通用
        specific: List = []
        generic: List = []
        for entry in entries:
            if goods_id is not None and entry.goods_id == goods_id:
                specific.append(entry)
            else:
                generic.append(entry)

        # 分别评分排序
        ranked_specific = self._score_entries(specific, query, goods_id, scene_key)
        ranked_generic = self._score_entries(generic, query, goods_id, scene_key)

        # 专属全部排在通用前面
        return ranked_specific + ranked_generic

    def _score_entries(
        self,
        entries: List,
        query: Optional[str],
        goods_id: Optional[int],
        scene_key: str = "",
    ) -> List[tuple]:
        """对一组条目评分并排序。返回 [(entry, score, match_type), ...]"""
        if not query or not query.strip():
            scored = []
            for entry in entries:
                score = int(entry.priority or 0) * 10
                scored.append((entry, score, "priority"))
            scored.sort(key=lambda x: -x[1])
            return scored

        query_clean = self._normalize_match_text(query)
        hints = self._query_intent_hints(query)
        scored: List[tuple] = []

        for entry in entries:
            score = 0
            match_type = "none"
            matched = False

            # 1. aliases 精确匹配（最高权重）
            alias_score = self._alias_match_score(query_clean, entry.aliases or "")
            if alias_score > 0:
                score += alias_score
                match_type = "alias_exact" if alias_score >= 200 else "alias_partial"
                matched = True

            # 2. 简单关键词匹配
            keyword_score = self._keyword_match_score(query, entry)
            if keyword_score > 0 and match_type == "none":
                score += keyword_score
                match_type = "keyword"
                matched = True
            elif keyword_score > 0:
                score += keyword_score
                matched = True

            # 3. 意图调整（boost/penalize）
            if hints:
                intent_adjustment = self._intent_score_adjustment(hints, entry, scene_key)
                score += intent_adjustment
                if intent_adjustment > 0:
                    matched = True

            if not matched:
                continue

            # 4. priority 只对已匹配条目加分，避免无关高优先级条目污染结果
            score += int(entry.priority or 0) * 10

            # 5. 向量检索 TODO
            # TODO: 接入 VectorRetriever，namespace=f"{scene_key}_knowledge"
            # vector_score = self.vector_retriever.rank(...)
            # if vector_score > 0:
            #     score += vector_score
            #     match_type = "vector"

            if score > 0:
                scored.append((entry, score, match_type))

        scored.sort(key=lambda x: -x[1])
        return scored

    @staticmethod
    def _knowledge_match_query(query: Optional[str]) -> str:
        """只用客户真实文本做知识匹配，避免商品卡片标题/价格污染检索。"""
        text = str(query or "").strip()
        marker = "客户消息："
        if marker not in text:
            return KnowledgeService._normalize_common_traditional(text)

        customer_part = text.split(marker, 1)[1]
        stop_markers = ("\n商品卡片：", "\n商品：", "\n订单信息：", "\n物流信息：")
        for stop in stop_markers:
            if stop in customer_part:
                customer_part = customer_part.split(stop, 1)[0]
        return KnowledgeService._normalize_common_traditional(customer_part.strip() or text)

    @staticmethod
    def _normalize_common_traditional(text: str) -> str:
        """Normalize common traditional Chinese terms seen in customer questions."""
        if not text:
            return text
        replacements = {
            "發": "发",
            "貨": "货",
            "遞": "递",
            "嗎": "吗",
            "幾": "几",
            "個": "个",
            "這": "这",
            "款": "款",
            "風": "风",
            "電": "电",
            "續": "续",
            "航": "航",
            "時": "时",
            "間": "间",
            "長": "长",
            "嗎": "吗",
            "麼": "么",
            "什麼": "什么",
            "沖": "冲",
            "滿": "满",
            "檔": "档",
            "顏": "颜",
            "色": "色",
            "質": "质",
            "保": "保",
            "開": "开",
            "關": "关",
            "聲": "声",
            "噪": "噪",
            "壞": "坏",
            "轉": "转",
            "葉": "叶",
            "繩": "绳",
            "無": "无",
            "帶": "带",
        }
        normalized = text
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized

    def _alias_match_score(self, query_clean: str, aliases: str) -> int:
        """aliases 匹配评分。"""
        if not query_clean or not aliases:
            return 0

        best_score = 0
        for alias in re.split(r"[/|;；\n\r]+", aliases):
            alias_clean = self._normalize_match_text(alias)
            if len(alias_clean) < 2:
                continue

            if alias_clean == query_clean:
                # 完全匹配
                best_score = max(best_score, 240 + min(len(alias_clean), 30))
            elif len(alias_clean) >= 4 and alias_clean in query_clean:
                # alias 是 query 的子串
                best_score = max(best_score, 115 + min(len(alias_clean), 12))
            elif len(query_clean) >= 6 and query_clean in alias_clean:
                # query 是 alias 的子串
                best_score = max(best_score, 80 + min(len(query_clean), 12))
            elif alias_clean in query_clean or query_clean in alias_clean:
                # 弱匹配
                best_score = max(best_score, 8 + min(len(alias_clean), 6))

        return best_score

    def _keyword_match_score(self, query: str, entry) -> int:
        """简单关键词匹配评分。"""
        words = self._search_terms(query)
        if not words:
            return 0

        texts = [
            entry.aliases or "",
            entry.answer or "",
            entry.section_title or "",
            entry.sub_intent or "",
        ]
        combined = " ".join(texts).lower()

        score = 0
        for word in words:
            word_lower = word.lower()
            if word_lower in combined:
                score += 4
            if word_lower in (entry.aliases or "").lower():
                score += 6
        return score

    # ── 意图识别 + 评分调整 ──────────────────────────────────────

    _LOGISTICS_QUERY_KW = ("快递", "物流", "包裹", "几小时到", "什么时候到", "到哪了", "到了吗", "发了吗", "寄出了", "还有多久到")
    _BATTERY_COMPLAINT_KW = ("没电", "半天", "用不了多久", "一会就没电", "不到一个小时", "续航短", "电不够用")
    _WRONG_MISSING_KW = ("发错货", "发错颜色", "颜色错", "少了", "少了一个", "少了个", "少发", "漏发", "缺件", "配件少")
    _NOTE_CHANGE_KW = ("备注一下", "备注发", "帮我改一下", "改一下地址", "能改地址", "更改收货地址", "改颜色", "换颜色")

    @classmethod
    def _query_intent_hints(cls, query: str) -> set:
        """识别 query 的意图标签集合，用于评分调整。"""
        q = str(query or "").strip()
        hints = set()
        # 物流/到货（排除"到了"这种出现在答案中的通用词）
        if any(kw in q for kw in cls._LOGISTICS_QUERY_KW):
            hints.add("logistics")
        # 售后续航投诉（区别于参数咨询）
        if any(kw in q for kw in cls._BATTERY_COMPLAINT_KW):
            hints.add("battery_complaint")
        # 错发/少件（不含"多少"）
        if any(kw in q for kw in cls._WRONG_MISSING_KW):
            hints.add("wrong_missing")
        # 备注/改地址/改颜色
        if any(kw in q for kw in cls._NOTE_CHANGE_KW):
            hints.add("note_change")
        return hints

    @classmethod
    def _intent_score_adjustment(cls, hints: set, entry, scene_key: str = "") -> int:
        """根据意图标签对条目进行加分/减分。返回调整值（可正可负）。"""
        section = (entry.section_title or "").lower()
        sub_intent = (entry.sub_intent or "").lower()
        answer = (entry.answer or "").lower()
        aliases = (entry.aliases or "").lower()
        combined = f"{section} {sub_intent} {answer} {aliases}"

        adj = 0

        # 物流查询 → 纯续航参数降权，物流类加分
        if "logistics" in hints:
            is_pure_battery = ("续航" in section or "续航" in sub_intent) and \
                              not any(kw in combined for kw in ("物流", "快递", "发货", "包裹", "配送"))
            is_logistics = any(kw in combined for kw in ("物流", "快递", "发货", "包裹", "配送"))
            if is_pure_battery:
                adj -= 60
            if is_logistics:
                adj += 30

        # 售后续航投诉 → 售后处理/转人工类加分（仅在售后场景）
        if "battery_complaint" in hints and scene_key == "aftersale":
            is_aftersale_handling = any(kw in combined for kw in ("转人工", "售后问题", "核实"))
            if is_aftersale_handling:
                adj += 80

        # 错发/少件 → 颜色参数类降权，错发/少件处理类（answer含转人工）加分
        if "wrong_missing" in hints:
            is_color_param = ("颜色" in section and "确认" in section)
            is_handling = "转人工" in answer
            is_wrong_section = any(kw in section for kw in ("发错", "少配件", "少件"))
            if is_color_param:
                adj -= 120
            if is_handling:
                adj += 60
            if is_wrong_section:
                adj += 80

        # 备注/改地址 → 修改处理类加分
        if "note_change" in hints:
            is_change_handling = any(kw in combined for kw in ("备注", "改地址", "修改", "更换"))
            if is_change_handling:
                adj += 40

        return adj

    def format_scene_results(self, results: List[Dict[str, Any]]) -> str:
        """将 search_scene_knowledge 结果格式化为 Agent 可读字符串。"""
        if not results:
            return "未找到相关知识。"

        parts = []
        for i, item in enumerate(results, 1):
            title = (item.get("section_title") or "").strip()
            sub_intent = item.get("sub_intent", "")
            answer = item.get("answer", "")
            score = item.get("score", 0)
            match_type = item.get("match_type", "")
            goods_id = item.get("goods_id")

            header = f"{i}. "
            if title:
                header += title
            elif sub_intent:
                header += sub_intent
            else:
                header += "命中知识"
            if goods_id:
                header += f" [商品{goods_id}]"
            header += f" (score={score}, {match_type})"

            parts.append(header)
            parts.append(f"  {answer}")
            parts.append("")

        return "\n".join(parts).strip()
