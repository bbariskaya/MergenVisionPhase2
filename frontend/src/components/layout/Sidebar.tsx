import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Upload,
  History,
  Users,
  Activity,
  Video,
} from 'lucide-react'
import styles from './sidebar.module.css'

const items = [
  { to: '/', icon: LayoutDashboard, label: 'Genel Bakış' },
  { to: '/videos/new', icon: Upload, label: 'Yeni Video Analizi' },
  { to: '/history', icon: History, label: 'İşlem Geçmişi' },
  { to: '/faces', icon: Users, label: 'Yüz Geçmişi' },
  { to: '/system', icon: Activity, label: 'Sistem Durumu' },
]

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <Video className={styles.brandIcon} aria-hidden="true" size={20} />
        <span className={styles.brandName}>MergenVision Video Intelligence</span>
      </div>
      <nav aria-label="Ana navigasyon">
        <ul className={styles.navList}>
          {items.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  `${styles.navLink} ${isActive ? styles.active : ''}`
                }
                end={item.to === '/'}
              >
                <item.icon size={18} aria-hidden="true" />
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  )
}
